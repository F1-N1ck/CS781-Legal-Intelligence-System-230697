
import json
import numpy as np


def head_tail_ids(ids, max_len, sep_id, head_frac=0.5):
    # Keep the first part and the last part of a long document.
    # ids = 1D token ids without special tokens. Output length <= max_len, ends with sep_id.
    ids = np.asarray(ids)
    body_len = max_len - 1
    n = len(ids)
    if n <= body_len:
        out = ids.tolist()
        out.append(int(sep_id))
        return out
    head_n = int(body_len * head_frac)
    tail_n = body_len - head_n
    out = []
    if head_n > 0:
        out.extend(ids[:head_n].tolist())
    if tail_n > 0:
        out.extend(ids[n - tail_n:].tolist())
    out.append(int(sep_id))
    return out


def chunk_starts(n_tokens, chunk_len, overlap):
    # Start offsets of overlapping chunks that cover the whole document.
    body_len = chunk_len - 1
    stride = body_len - overlap
    if stride < 1:
        raise ValueError("overlap must be smaller than chunk_len - 1")
    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + body_len >= n_tokens:
            break
        start = start + stride
    return starts


def pick_starts(starts, max_chunks, pick):
    # Reduce too many chunks to max_chunks.
    # pick: first | last | head_tail | uniform
    total = len(starts)
    if max_chunks is None or total <= max_chunks:
        return starts
    if pick == "first":
        return starts[:max_chunks]
    if pick == "last":
        return starts[total - max_chunks:]
    if pick == "head_tail":
        n_head = max_chunks // 2
        n_tail = max_chunks - n_head
        return starts[:n_head] + starts[total - n_tail:]
    if pick == "uniform":
        positions = np.linspace(0, total - 1, max_chunks)
        chosen = []
        for p in positions:
            idx = int(round(float(p)))
            if idx not in chosen:
                chosen.append(idx)
        result = []
        for idx in chosen:
            result.append(starts[idx])
        return result
    raise ValueError("unknown pick: " + str(pick))


def make_chunks(ids, sep_id, chunk_len, overlap, max_chunks=None, pick="uniform"):
    # Returns (chunks, rel_pos). Each chunk ends with sep_id.
    # rel_pos = where the chunk starts inside the document, 0..1 (use as a position feature).
    ids = np.asarray(ids)
    n = len(ids)
    body_len = chunk_len - 1
    starts = chunk_starts(n, chunk_len, overlap)
    starts = pick_starts(starts, max_chunks, pick)
    chunks = []
    rel_pos = []
    for s in starts:
        piece = ids[s:s + body_len].tolist()
        piece.append(int(sep_id))
        chunks.append(piece)
        rel_pos.append(s / max(1, n))
    return chunks, rel_pos


def pad_right(seqs, pad_id):
    # Right-pad a list of id lists. Returns input_ids [B, T], attention_mask [B, T], lengths [B].
    lengths = []
    for s in seqs:
        lengths.append(len(s))
    max_len = max(lengths)
    batch = len(seqs)
    input_ids = np.full((batch, max_len), int(pad_id), dtype=np.int64)
    mask = np.zeros((batch, max_len), dtype=np.int64)
    for i in range(batch):
        L = lengths[i]
        input_ids[i, :L] = np.asarray(seqs[i], dtype=np.int64)
        mask[i, :L] = 1
    return input_ids, mask, np.asarray(lengths, dtype=np.int64)


def multi_hot(label_lists, statute_ids):
    # label_lists: list of lists of statute IDs. statute_ids: ordered list of all statute IDs.
    # Returns a [N, S] int array, 1 where the statute is relevant.
    index_of = {}
    for j in range(len(statute_ids)):
        index_of[int(statute_ids[j])] = j
    Y = np.zeros((len(label_lists), len(statute_ids)), dtype=np.int64)
    for i in range(len(label_lists)):
        for s in label_lists[i]:
            if int(s) in index_of:
                Y[i, index_of[int(s)]] = 1
    return Y


def clean_prediction_ids(raw_ids, allowed_ids):
    # Keep only valid statute IDs, as unique integers. Order is kept (not a ranking).
    allowed = set(int(a) for a in allowed_ids)
    seen = set()
    out = []
    for x in raw_ids:
        v = int(x)
        if v in allowed and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def macro_micro_f1(Y_true, Y_pred):
    # Y_true, Y_pred: [N, S] 0/1 arrays. Returns (macro_f1, micro_f1, subset_accuracy).
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)
    f1_list = []
    tp_all = 0
    fp_all = 0
    fn_all = 0
    for s in range(Y_true.shape[1]):
        tp = int(((Y_true[:, s] == 1) & (Y_pred[:, s] == 1)).sum())
        fp = int(((Y_true[:, s] == 0) & (Y_pred[:, s] == 1)).sum())
        fn = int(((Y_true[:, s] == 1) & (Y_pred[:, s] == 0)).sum())
        tp_all += tp
        fp_all += fp
        fn_all += fn
        if tp == 0:
            f1_list.append(0.0)
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1_list.append(2 * precision * recall / (precision + recall))
    macro = float(sum(f1_list) / max(1, len(f1_list)))
    if tp_all == 0:
        micro = 0.0
    else:
        p = tp_all / (tp_all + fp_all)
        r = tp_all / (tp_all + fn_all)
        micro = float(2 * p * r / (p + r))
    subset_acc = float((Y_true == Y_pred).all(axis=1).mean())
    return macro, micro, subset_acc


def best_global_threshold(probs, Y_true):
    # Scan one threshold for all statutes. Returns (best_threshold, macro_f1).
    probs = np.asarray(probs)
    best_t = 0.5
    best_f = -1.0
    for t in np.arange(0.05, 0.951, 0.01):
        pred = (probs >= t).astype(int)
        macro, _, _ = macro_micro_f1(Y_true, pred)
        if macro > best_f:
            best_f = macro
            best_t = float(t)
    return best_t, best_f


def write_submission(doc_ids, pred_lists, allowed_ids, path):
    # Codabench format: JSON list of {"id": ..., "pred": [unique int statute IDs]}, one per test doc.
    if len(doc_ids) != len(pred_lists):
        raise ValueError("doc_ids and pred_lists differ in length")
    rows = []
    for i in range(len(doc_ids)):
        clean_pred = clean_prediction_ids(pred_lists[i], allowed_ids)
        rows.append({"id": doc_ids[i], "pred": clean_pred})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    print("wrote", len(rows), "predictions to", path)
