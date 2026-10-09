"""Steps 4 and 6: melanoma %, predictive entropy and mutual information per method.

For every validation and test photo:
- Baseline: 1 forward pass.
- MC Dropout: 30 passes of the baseline with dropout on.
- Deep Ensemble: 5 models, 1 pass each.
Writes one predictions file per method.
"""
