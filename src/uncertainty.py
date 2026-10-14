"""Uncertainty from several answers per photo: mean %, predictive entropy, mutual information.

Shared by predict.py (GPU) and evaluate.py (no GPU), so it only needs numpy.
"""
import numpy as np

EPS = 1e-7


def binary_entropy(p):
    """Entropy (nats) of a melanoma probability: 0 when sure, ln 2 ~ 0.693 at 50%."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def uncertainty(samples):
    """samples (N, S): S answers per photo -> mean %, predictive entropy, mutual information.

    predictive entropy = entropy of the mean answer (total uncertainty)
    mutual information = predictive entropy - mean entropy of single answers (disagreement)
    """
    p_mean = samples.mean(axis=1)
    entropy = binary_entropy(p_mean)
    mutual_info = entropy - binary_entropy(samples).mean(axis=1)
    return p_mean, entropy, np.maximum(mutual_info, 0.0)
