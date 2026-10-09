"""Steps 3 and 5: train one model (baseline = seed 0, ensemble = seeds 1-4).

Usage: python src/train.py --config configs/base.yaml --seed 0

- Epoch loop, validation AUC each epoch, keep the best checkpoint, early stop.
- Saves config, seed, metrics and training time to results/<run>/.
- Checkpoint goes to checkpoints/ (not in git).
"""
