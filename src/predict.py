"""Steps 4 and 6: melanoma %, predictive entropy and mutual information per method.

Usage: python src/predict.py --config configs/base.yaml [--methods mc_stochdepth ...]

For every validation and test photo:
- Baseline: seed 0 model, 1 forward pass.
- MC Dropout: seed 0 model, 30 passes with dropout on.
- Deep Ensemble: seeds 0-4, 1 pass each.
- Ensemble + MC Dropout: seeds 0-4, 30 passes each with dropout on (150 answers).
- MC Dropout + StochasticDepth: seeds 0-4, 30 passes each with dropout and block skipping on
  (added after the main evaluation; saved per seed in mc_stochdepth_samples.npy, columns 30i..30i+29).
Writes results/predictions/<method>.csv (one row per photo), <method>_samples.npy
(every individual answer, rows aligned with the csv) and timing.json. Only the methods asked for
are recomputed; other files and timings are left untouched.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data import get_loaders, load_split
from model import build_model, enable_mc_dropout
from uncertainty import uncertainty

METHODS = ["baseline", "mc_dropout", "ensemble", "ensemble_mc", "mc_stochdepth"]


def load_model(seed, cfg, device):
    model = build_model(cfg["model"]["dropout"], pretrained=False)
    model.load_state_dict(torch.load(f"checkpoints/seed{seed}.pt", map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def probabilities(model, loader, device):
    """One pass over the loader -> melanoma probability per photo, in loader order."""
    out = []
    for images, _, _ in loader:
        out.append(torch.sigmoid(model(images.to(device, non_blocking=True)).view(-1)).cpu())
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser(description="Save per-photo predictions for each method")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS[:4])
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg["train"]["batch_size"] = 256  # inference only: bigger batches, same results
    device = "cuda"
    torch.manual_seed(0)  # fixes which neurons MC Dropout switches off

    loaders = get_loaders(cfg, splits=("val", "test"))
    split, _ = load_split(cfg)
    rows = split[split["split"].isin(["val", "test"])]
    rows = rows.sort_values("split", key=lambda s: s.map({"val": 0, "test": 1}), kind="stable")
    rows = rows.reset_index(drop=True)  # same order as the val loader followed by the test loader
    loader_labels = np.concatenate([loaders[s].dataset.labels.numpy() for s in ("val", "test")])
    assert np.array_equal(loader_labels, rows["label"].values), "loader order does not match split.csv"

    def run_all(model, passes=1):
        """(N_val + N_test, passes) probabilities."""
        return np.stack([
            np.concatenate([probabilities(model, loaders[s], device) for s in ("val", "test")])
            for _ in range(passes)
        ], axis=1)

    u = cfg["uncertainty"]
    methods, timing = {}, {}

    seeds = u["ensemble_seeds"]

    def mc(stochastic_depth):
        """Every seed, 30 dropout passes each -> (N, 5 * 30), seed i in columns 30i..30i+29."""
        return np.concatenate([
            run_all(enable_mc_dropout(load_model(seed, cfg, device), stochastic_depth), passes=u["mc_passes"])
            for seed in seeds], axis=1)

    runs = {
        "baseline": lambda: run_all(load_model(0, cfg, device)),
        "mc_dropout": lambda: run_all(enable_mc_dropout(load_model(0, cfg, device)), passes=u["mc_passes"]),
        "ensemble": lambda: np.concatenate([run_all(load_model(s, cfg, device)) for s in seeds], axis=1),
        "ensemble_mc": lambda: mc(stochastic_depth=False),
        "mc_stochdepth": lambda: mc(stochastic_depth=True),
    }
    for name in args.methods:
        start = time.time()
        methods[name] = runs[name]()
        timing[name] = time.time() - start

    out_dir = Path("results/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, samples in methods.items():
        p, entropy, mi = uncertainty(samples)
        table = rows[["image_id", "lesion_id", "dx", "label", "split"]].copy()
        table["p_melanoma"], table["entropy"], table["mutual_info"] = p, entropy, mi
        table.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6g")
        np.save(out_dir / f"{name}_samples.npy", samples.astype(np.float32))
        val = table[table["split"] == "val"]
        print(f"{name:11s} answers/photo {samples.shape[1]:2d}  "
              f"val mean entropy {val['entropy'].mean():.4f}  val mean MI {val['mutual_info'].mean():.4f}  "
              f"{timing[name]:.1f}s")

    timing_file = out_dir / "timing.json"
    saved = json.load(open(timing_file)) if timing_file.exists() else {}
    saved.update({k: round(v, 2) for k, v in timing.items()})
    saved["photos"] = len(rows)
    gpu = torch.cuda.get_device_name()
    if set(args.methods) == set(METHODS[:4]):
        saved["gpu"] = gpu
    for name in args.methods:
        saved.pop(f"gpu_{name}", None)
        if "gpu" in saved and saved["gpu"] != gpu:
            saved[f"gpu_{name}"] = gpu  # timed on a different GPU than the rest
    json.dump(saved, open(timing_file, "w"), indent=2)


if __name__ == "__main__":
    main()
