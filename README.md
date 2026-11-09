# CS781 Capstone: In-House Legal Intelligence System

**Nikhil N (230697), IIT Kanpur** | CS781 (Large Language Models), 2026

A seven-phase pipeline built on legal language models that we **pretrained from scratch** (Phase 1) and then adapted to six legal tasks with **hand-written parameter-efficient fine-tuning** (LoRA, LoRMA, adapters, distillation; no `peft` library).

- Report: [`report/CS781-CP-Nikhil-N-230697-report.pdf`](report/CS781-CP-Nikhil-N-230697-report.pdf) (LaTeX source in `report/`)
- Dataset: `Exploration-Lab/CS-781-Capstone` (gated Hugging Face dataset, not included here)
- Compute: Kaggle, 2x T4 GPUs

## Results (Codabench, test set)

| Phase | Task | Metric | Best score | Best system | Rank |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining | BERTScore F1 | 0.8317 | Model 4 (GQA/MHA + MoE) | 65 |
| 2 | Legal NER | span F1 | 0.5929 | Model 4 + LoRMA | 17 |
| 3 | Rhetorical roles | macro-F1 | 0.571 | Model 4 + LoRMA | 20 |
| 4 | Court judgment prediction | macro-F1 | 0.6879 | Model 4 + LoRA | 86 |
| 5 | Statute identification | macro-F1 | 0.3713 | Model 1 + LoRA | 72 |
| 6 | Summarization | BERTScore F1 | 0.8036 | Model 4 + LoRA | 122 |
| 7 | Prior case retrieval | micro-F1@7 | 0.3237 | Model 4 + LoRA | 93 |

37 valid Codabench submissions in total. Ranks recorded on 9 Nov 2026.

## Phase-1 backbones

| Model | Architecture | Params | Outcome |
|---|---|---|---|
| 1 | GQA + RoPE | 138.25M | Trained on ~1.7B tokens, val perplexity 8.7 |
| 2 | MLA + sparse attention + MoE | 221.57M | ~0.95B tokens, val perplexity 46.75 |
| 3 | GQA + RetNet | 164.87M | Training diverged (NaN gradients), not completed |
| 4 | GQA/MHA + MoE | 252.99M | Best model, val perplexity 7.84, BERTScore 0.8317 |

All models share a custom 50k byte-level BPE tokenizer trained on the legal corpus.

## Repository layout

```
code/
  phase1/   tokenizer, Model 1-4 pretraining, generation, test preparation
  phase2/   Legal NER
  phase3/   Rhetorical roles
  phase4/   Court judgment prediction (CJPE)
  phase5/   Legal statute identification (LSI)
  phase6/   Summarization
  phase7/   Prior case retrieval (PCR)
submissions/  files uploaded to Codabench, one folder per phase
report/       LaTeX source, bibliography, figures, compiled PDF
```

Notebooks are numbered in the order they are meant to be run. `M1` = Model 1, `M4` = Model 4, `kd` = knowledge distillation.

### Phase 1: pretraining (`code/phase1/`)

| Notebook | Purpose |
|---|---|
| `00_starter_code` | Course starter code |
| `01_tokenizer_and_corpus_tokenization` | Train 50k BPE tokenizer, tokenize corpus into `.npy` shards (reconstructed, see Notes) |
| `02_model1_gqa_rope_train` | Model 1 pretraining |
| `03_model1_calibration` | Model 1 calibration / decoding checks |
| `04_model1_generate` | Model 1 test-set generation |
| `05` to `09_model2_*` | Model 2 (MLA + DSA + MoE) training runs (v1, v2, v3, v8, project notebook) |
| `10_model2_generate` | Model 2 test-set generation |
| `11_model3_gqa_retnet_train_diverged` | Model 3 attempt (NaN gradients) |
| `12_model4_gqa_mha_moe_train`, `13_..._continued` | Model 4 pretraining and continued run |
| `14_model4_generate` | Model 4 test-set generation |
| `15_prepare_test_v1`, `16_prepare_test_v2` | Test-set preparation |

### Phases 2 to 7 (`code/phase2` ... `code/phase7`)

Each phase starts with a `00_*_data_prep` notebook that cleans and tokenizes the task data and writes a helper module (`*_utils.py`, embedded in the notebook) used by all later notebooks of that phase.

| Phase | Data prep | Runs |
|---|---|---|
| 2 NER | `00_ner_data_prep` | M1 LoRA, M4 LoRA, M4 LoRMA, M2 DSA full fine-tune, KD (M4 to M1), ensemble |
| 3 Rhetorical roles | `00_rr_data_prep` | M1 LoRA (+v2), M1 LoRMA, M1 LoRA+KD, M4 LoRA, M4 LoRMA, `07_rr_generate_submission` |
| 4 CJPE | `00_cjpe_data_prep` | M1/M4 x LoRA/LoRMA |
| 5 LSI | `00_lsi_clean_tokenize_analysis`, `01_lsi_data_prep`, `lsi_utils.py` | M1/M4 x LoRA/LoRMA, KD (M1), ensemble |
| 6 Summarization | `00_summ_data_prep` | M1/M4 x LoRA/LoRMA, generation notebooks for M4, KD |
| 7 Prior case retrieval | `00_pcr_data_prep` | M1/M4 x LoRA/LoRMA, KD, ensemble (BM25 + neural fusion) |

## Methods in one paragraph

LoRA and LoRMA adapters are implemented by hand (including a LoRMA "plus" variant that fixed the "perm" variant failing on Model 1 in Phase 4). Task heads are BiGRUs; retrieval uses BM25 fused with a neural score by dual-softmax; summarization uses MBR decoding; multi-label tasks use out-of-fold thresholds. Phase-1 decoding: nucleus sampling (top-p 0.95, T 0.7), trigram blocking, stop at 128 words.

## How to run

All notebooks were run on Kaggle (T4 x2). Install dependencies with `pip install -r requirements.txt` if running elsewhere.

1. Store your Hugging Face token in Kaggle Secrets as `HF_TOKEN` (needed to download the dataset). No tokens are stored in this repo.
2. Run `code/phase1` notebooks to produce the tokenizer and the model folders.
3. For each downstream phase, run its `00_*_data_prep` notebook first, then attach its output, the Phase-1 model folder and the model `.py` files to the training notebook.
4. Each training notebook has a `SMOKE_TEST` switch for a quick end-to-end check.

## Submissions

See [`submissions/README.md`](submissions/README.md) for what each file is.

## Notes

- Notebooks are committed without outputs. Model weights, tokenized data and checkpoints are not committed.
- `01_tokenizer_and_corpus_tokenization.ipynb` was reconstructed from the original cells after the original file was lost; the sharded tokenization logic is the original, the surrounding config/download cells were reassembled.
- Phase 7 distillation and ensemble were evaluated on dev data only and not submitted to Codabench.
- Phase 4 Model 1 LoRMA-"perm" scored 0.4837; its submission zip is not included.
