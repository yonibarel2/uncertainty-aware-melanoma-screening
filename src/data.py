"""Step 2: data loading, binary label and lesion_id split.

- Reads HAM10000 metadata, labels melanoma = 1, everything else = 0.
- Splits by lesion_id into train / val / test (70 / 15 / 15), ~11% melanoma in each.
- Saves the split to data/split.csv so every run uses the same split.
- Dataset: resize to 224, augmentation on train only.
"""
