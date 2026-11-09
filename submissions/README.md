# Submissions

Files uploaded to Codabench, one folder per phase. Naming: `m1`/`m4` = backbone model, `lora`/`lorma` = adapter type, `kd` = distillation.

| Phase | Best file | Score | Other files |
|---|---|---|---|
| 1 | `model4_prediction_a.json` / `_b.json` / `submission_phase1.zip` | 0.8317 (Model 4) | `model1_`, `model2_`, `model3_`, `prediction_early_run.json` |
| 2 | `m4_lorma.zip` | 0.5929 | `m1_lora`, `m4_lora`, `m4_lorma_perm`, `m2_dsa_full_finetune`, `kd_m4_to_m1`, `ensemble` |
| 3 | `m4_lorma.zip` | 0.571 | `m1_lora`, `m1_lorma`, `m1_kd`, `m4_lora` |
| 4 | `m4_lora.zip` | 0.6879 | `m1_lora`, `m1_lora_attn`, `m1_lorma_plus`, `m4_lorma` |
| 5 | `m1_lora_submission.json` | 0.3713 | `m1_lorma` (0.3660), `m4_lorma` (0.3659), `m4_lora` (0.3640), `kd_m1.zip`; `*_thresholds.json` and `*_run_summary.json` are per-run logs |
| 6 | `m4_lora.zip` | 0.8036 | `m1_lora`, `m1_lorma`, `m4_lorma` |
| 7 | `m4_lora.zip` | 0.3237 | `m4_lorma` (0.3213), `m1_lora` (0.3231), `m1_lorma` (0.3184) |
