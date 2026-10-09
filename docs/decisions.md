# Decision Log

One entry per key decision: what we chose, what else we considered, and why. This is what we defend in the presentation.

| Date | Decision | Alternatives considered | Why |
| --- | --- | --- | --- |
| 2026-10-09 | CNN: EfficientNet-B0 | ResNet-50 | ~5× smaller and faster (matters with 5 ensemble models), similar accuracy, dropout built in for MC Dropout |
| 2026-10-09 | Compute: RunPod rented GPU | Colab, local machine | Fast, cheap (budget up to $100), works with VS Code over SSH |
| 2026-10-09 | Data source: Kaggle (HAM10000) | Harvard Dataverse, ISIC archive | One download command with an API key |
| 2026-10-09 | Split: all 10,015 photos, by `lesion_id`, 70/15/15 | Image-level split; one photo per lesion | No leakage, keeps the 11% melanoma share, matches the proposal |
| 2026-10-09 | Task: binary only (melanoma vs. rest) | Also the 7-class problem | Time; 7-class is future work |
| 2026-10-09 | Trust score: predictive entropy | Mutual information; max softmax % | Entropy already includes MI; MI is reported separately for analysis |
