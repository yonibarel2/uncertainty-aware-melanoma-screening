"""Step 3: EfficientNet-B0 and focal loss.

- EfficientNet-B0 with ImageNet weights, last layer replaced by a single melanoma logit.
- A switch to keep dropout on at prediction time (MC Dropout).
- Binary focal loss (gamma, alpha from the config).
"""
