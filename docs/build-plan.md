# Build and Run Plan

> Mirrored from the Claude Doc "Build and Run Plan" (approved 2026-10-09). This is the plan Claude Code works from.

The plan to build, run and write up the project by 15.10.2026.

## Decisions made

| Decision | Choice | Why |
| --- | --- | --- |
| CNN | EfficientNet-B0 | ~5× smaller and faster than ResNet-50 (matters with 5 ensemble models), similar accuracy, dropout built in for MC Dropout |
| Compute | RunPod, rented GPU (e.g. RTX 4090) | Fast, cheap (budget up to $100), works with VS Code over SSH |
| Data source | Kaggle (HAM10000) | Fastest and simplest: one download command with an API key |
| Data split | Keep all 10,015 photos, split by `lesion_id` | No leakage, keeps the 11% melanoma share, matches the proposal |
| Task | Binary only (melanoma vs. rest) | The 7-type problem is future work |
| Trust score | Predictive entropy | Already includes mutual information; MI is reported separately for analysis only |

## The steps

Each step ends with something saved in the repo, so every number in the report traces back to a run.

1. **Setup:** rent the RunPod GPU, connect VS Code over SSH, create the repo folders, install PyTorch.
   - Output: an environment that runs a test script on the GPU.
2. **Data + split:** download HAM10000, make the binary label, split by `lesion_id` into train 70% / validation 15% / test 15%, ~11% melanoma in each.
   - Output: a saved split file, so every run uses the same split.
3. **Baseline:** fine-tune EfficientNet-B0 with focal loss, keeping the best epoch on validation.
   - Output: model checkpoint + validation metrics.
4. **MC Dropout:** no new training. The baseline asked 30 times with dropout on.
5. **Deep Ensemble:** train 4 more copies with different seeds. With the baseline, that makes 5. Optional, only once steps 1–8 work: Ensemble + MC Dropout combined, each of the 5 models asked 30 times (150 answers per photo). No extra training, one more row in the comparison.
6. **Trust scores:** for every validation and test photo, save the %, entropy and mutual information for each method.
   - Output: one predictions file per method.
7. **Pick the cutoff (validation) → (b):** cost table, missed melanoma = 100, doctor visit = 1. Also try other ratios (10, 50) to show how the cutoff moves. The cutoff is set on predictive entropy.
8. **Final exam (test) → (a) + (d):** sensitivity, AUC, ECE, Brier, the 2 curves, training time, for all 3 methods, plus a reliability diagram per method (where the % is honest and where it lies). Optional: fix the % with histogram binning or temperature scaling, fitted on validation. It makes the % honest, but can't find which single 99% is wrong.
   - Output: results table + figures. Also check that most healthy photos get a high % (above ~95%), which confirms the focal-loss claim that they are "easy".
9. **Failure analysis (test) → (c):** uncertainty and errors by true lesion type, plus a few example photos.
10. **Write:** report (max 5 pages) and slides, from the saved figures.

## Settings

Starting values. We change one only if validation results say so, and log why in `docs/decisions.md`.

| Setting | Value | Why |
| --- | --- | --- |
| Image size | 224 × 224 | EfficientNet-B0's native input size |
| Augmentation | random flips, rotation, color jitter (train only) | more variety, less overfitting |
| Optimizer | AdamW, learning rate 1e-4 | small steps so fine-tuning keeps ImageNet knowledge |
| Batch size | 32 | standard, fits GPU memory easily |
| Epochs | up to 20, keep the best on validation AUC, stop if no gain for 5 epochs | early stopping prevents overfitting |
| Focal loss | γ = 2, α = 0.75 for melanoma | focal loss paper defaults, α tilted toward rare melanoma |
| Dropout | 0.2 | EfficientNet-B0 default |
| MC Dropout passes | T = 30 | as in the proposal |
| Ensemble size | M = 5 (seeds 0–4) | as in the proposal |
| ECE bins | 15 | common choice |
| Costs | missed melanoma 100, doctor visit 1 (also 10 and 50) | a miss is far worse; the range shows sensitivity to that guess |

## Repo structure

One script per step, results saved per run. A run = one training of one model (~5 total: baseline seed 0 + ensemble seeds 1–4).

| Path | Holds | Notes |
| --- | --- | --- |
| `configs/` | settings files | 1 shared config + seed as an option, e.g. `train.py --config base.yaml --seed 3`. Each setting has a short note inside the YAML + a section in the main README; no separate README. |
| `src/data.py` | loading, binary label, split by `lesion_id` | reads metadata, labels melanoma = 1, saves `split.csv`, feeds photos (resize 224, augmentation on train only) |
| `src/model.py` | EfficientNet-B0 + focal loss | ImageNet weights, last layer → 1 output; switch to keep dropout on at prediction (MC Dropout) |
| `src/train.py` | training (baseline and each ensemble seed) | epochs loop, validation AUC each epoch, best checkpoint, early stop; saves config, seed, metrics, training time |
| `src/predict.py` | %, entropy, mutual information per method | for validation and test photos: baseline (1 ask), MC Dropout (30 asks), Ensemble (5 models); one predictions file per method |
| `src/evaluate.py` | metrics, curves, cost table, failure analysis | reads predictions only; cutoff on validation, all metrics + figures on test |
| `results/<run>/` | config, seed, metrics, predictions (in git) | every number in the report points to a folder here |
| `figures/` | every plot used in the report and slides (in git) | generated by code only, never made by hand |
| `docs/decisions.md` | decision log, one line each | |
| `data/`, `checkpoints/` | photos and model weights (**not** in git) | |
| `README.md` | how to run everything, step by step | |
