"""Steps 3 and 5: train one model (baseline = seed 0, ensemble = seeds 1-4).

Usage: python src/train.py --config configs/base.yaml --seed 0

- Epoch loop, validation AUC each epoch, keep the best checkpoint, early stop.
- Saves config, seed, per-epoch history, metrics and training time to results/seed<N>/.
- Checkpoint goes to checkpoints/seed<N>.pt (not in git).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from data import get_loaders
from model import FocalLoss, build_model


def set_seed(seed):
    """Same seed -> same initial head weights, same shuffling and augmentation order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total, count = 0.0, 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Validation loss and AUC (dropout off)."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            all_logits.append(model(images).float().view(-1).cpu())
        all_labels.append(labels)
    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    loss = loss_fn(logits, labels).item()
    auc = roc_auc_score(labels.numpy(), torch.sigmoid(logits).numpy())
    return loss, auc


def main():
    parser = argparse.ArgumentParser(description="Train one EfficientNet-B0 with focal loss")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, help="override config (e.g. 1 for a quick test)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["seed"] = args.seed
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs
    t = cfg["train"]

    run = f"seed{args.seed}"
    run_dir, ckpt_dir = Path("results") / run, Path("checkpoints")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{run}.pt"
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)

    set_seed(args.seed)
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    loaders = get_loaders(cfg, splits=("train", "val"))
    model = build_model(cfg["model"]["dropout"]).to(device).to(memory_format=torch.channels_last)
    loss_fn = FocalLoss(cfg["loss"]["gamma"], cfg["loss"]["alpha"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"])

    history, best_auc, best_epoch = [], -1.0, 0
    start = time.time()
    for epoch in range(1, t["max_epochs"] + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
        val_loss, val_auc = evaluate(model, loaders["val"], loss_fn, device)
        improved = val_auc > best_auc
        if improved:
            best_auc, best_epoch = val_auc, epoch
            torch.save(model.state_dict(), ckpt_path)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_auc": val_auc, "seconds": round(time.time() - epoch_start, 1)})
        print(f"[{run}] epoch {epoch:2d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
              f"val_auc {val_auc:.4f}  {history[-1]['seconds']}s{'  *best' if improved else ''}", flush=True)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if epoch - best_epoch >= t["early_stop_patience"]:
            print(f"[{run}] early stop: no val AUC gain for {t['early_stop_patience']} epochs")
            break

    metrics = {
        "run": run,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_auc": round(best_auc, 4),
        "epochs_run": len(history),
        "train_time_sec": round(time.time() - start, 1),
        "gpu": torch.cuda.get_device_name(),
        "checkpoint": str(ckpt_path),
    }
    json.dump(metrics, open(run_dir / "metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
