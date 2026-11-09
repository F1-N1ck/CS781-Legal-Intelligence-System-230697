import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/claude/cs781report/figs/"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREY = "#718096"

# ---- Fig: statute long tail (Phase 5, numbers from the data-prep log) ----
top_ids = [5, 34, 97, 43, 39, 12, 13, 10, 77, 94]
top_cnt = [11522, 10284, 7773, 7617, 7015, 4267, 4081, 4057, 3968, 3797]
low_ids = [84, 25, 23, 92, 66, 59, 50, 74, 4, 27]
low_cnt = [267, 286, 303, 306, 307, 308, 317, 319, 324, 329]
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
ax[0].bar([str(i) for i in top_ids], top_cnt, color=BLUE)
ax[0].set_title("10 most frequent statutes", fontsize=9)
ax[0].set_xlabel("statute id")
ax[0].set_ylabel("train documents")
ax[1].bar([str(i) for i in low_ids], low_cnt, color=ORANGE)
ax[1].axhline(334, color=GREY, ls="--", lw=1)
ax[1].text(9.4, 340, "rare cut-off (334)", ha="right", va="bottom", fontsize=7.5, color=GREY)
ax[1].set_title("10 least frequent statutes", fontsize=9)
ax[1].set_xlabel("statute id")
ax[1].set_ylim(0, 420)
plt.tight_layout()
plt.savefig(OUT + "fig_statutes.pdf")
plt.close()

# ---- Fig: PEFT comparison, Phases 2 and 3 (Codabench primary metric) ----
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
labels = ["M1\nLoRA", "M1\nLoRMA", "M4\nLoRA", "M4\nLoRMA"]
p3 = [0.518, 0.538, 0.537, 0.571]
cols = [BLUE, ORANGE, BLUE, ORANGE]
b = ax[0].bar(labels, p3, color=cols)
ax[0].set_ylim(0.4, 0.62)
ax[0].set_title("Phase 3: rhetorical roles (macro-F1)", fontsize=9)
for r, v in zip(b, p3):
    ax[0].text(r.get_x() + r.get_width() / 2, v + 0.003, "%.3f" % v, ha="center", fontsize=7.5)
p2 = [0.5906, None, 0.5661, 0.5929]
xs = [0, 2, 3]
vals = [0.5906, 0.5661, 0.5929]
b2 = ax[1].bar(xs, vals, color=[BLUE, BLUE, ORANGE])
ax[1].set_xticks(range(4))
ax[1].set_xticklabels(labels)
ax[1].set_ylim(0.5, 0.62)
ax[1].set_title("Phase 2: legal NER (span-F1)", fontsize=9)
for x, v in zip(xs, vals):
    ax[1].text(x, v + 0.002, "%.3f" % v, ha="center", fontsize=7.5)
ax[1].text(1, 0.505, "not run", ha="center", fontsize=7.5, color=GREY)
plt.tight_layout()
plt.savefig(OUT + "fig_peft.pdf")
plt.close()

# ---- Fig: PCR dev F1@7 ----
fig, ax = plt.subplots(figsize=(3.3, 2.3))
names = ["BM25\nhead+tail", "BM25\nfull text", "frozen\nM1", "M4\nLoRMA", "M4\nLoRA"]
vals = [0.32, 0.19, 0.2424, 0.3024, 0.3180]
cols = [GREY, GREY, BLUE, ORANGE, ORANGE]
b = ax.bar(names, vals, color=cols)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.006, "%.3f" % v, ha="center", fontsize=7.5)
ax.set_ylabel("dev F1@7")
ax.set_ylim(0, 0.38)
ax.tick_params(axis="x", labelsize=7.5)
plt.tight_layout()
plt.savefig(OUT + "fig_pcr.pdf")
plt.close()

# ---- Fig: Phase 1 Codabench metrics, M1 vs M2 ----
fig, ax = plt.subplots(figsize=(3.3, 2.3))
metrics = ["BERTScore", "ROUGE-L", "BLEU-4"]
m1 = [0.8305, 0.184, 0.0226]
m2 = [0.8115, 0.176, 0.0160]
rel = [b / a for a, b in zip(m1, m2)]
ax.bar(metrics, [1.0] * 3, color=BLUE, width=0.38, align="edge", label="Model 1 (1.7B tok)")
ax.bar([x + 0.38 for x in range(3)], rel, color=ORANGE, width=0.38, align="edge", label="Model 2 (0.95B tok)")
ax.set_xticks([x + 0.38 for x in range(3)])
ax.set_xticklabels(metrics)
ax.set_ylabel("score relative to Model 1")
ax.set_ylim(0, 1.45)
for i, v in enumerate(rel):
    ax.text(i + 0.57, v + 0.02, "%.2f" % v, ha="center", fontsize=7.5)
ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=1)
plt.tight_layout()
plt.savefig(OUT + "fig_phase1.pdf")
plt.close()
print("ok")
