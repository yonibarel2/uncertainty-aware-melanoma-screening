"""Steps 7-9: cutoff, metrics, curves, failure analysis. Reads predictions files only.

- Step 7 (validation): cost-based defer cutoff on predictive entropy (miss = 100, visit = 1; also 10, 50).
- Step 8 (test): sensitivity, AUC, ECE, Brier, risk-coverage and sensitivity-coverage curves,
  reliability diagrams, timing.
- Step 9 (test): uncertainty and errors by true lesion type, example photos.
All figures are saved to figures/.
"""
