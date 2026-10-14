"""Step 2: data loading, binary label and lesion_id split.

- Reads HAM10000 metadata, labels melanoma = 1, everything else = 0.
- Splits by lesion_id into train / val / test (70 / 15 / 15), stratified so each has ~11% melanoma.
- Saves the split to results/split.csv (in git) so every run uses the same split.
- Resizes every photo to 224 x 224 once and caches them as one uint8 array (fast loading).
- Dataset: augmentation on train only.

Usage (run once): python src/data.py --config configs/base.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

IMAGE_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_metadata(root):
    """One row per image: image_id, lesion_id, dx, label (melanoma = 1), path."""
    root = Path(root)
    meta = pd.read_csv(root / "HAM10000_metadata.csv")
    paths = {p.stem: p for d in IMAGE_DIRS for p in (root / d).glob("*.jpg")}
    meta["path"] = meta["image_id"].map(paths)
    missing = meta["path"].isna().sum()
    assert missing == 0, f"{missing} images listed in metadata were not found"
    meta["label"] = (meta["dx"] == "mel").astype(int)
    return meta


def make_split(meta, fractions, seed):
    """Assign each lesion (and so all its images) to train / val / test, stratified by label."""
    lesions = meta.groupby("lesion_id")["label"].max().reset_index()
    train_frac, val_frac, test_frac = fractions
    train, rest = train_test_split(
        lesions, train_size=train_frac, stratify=lesions["label"], random_state=seed
    )
    val, test = train_test_split(
        rest, train_size=val_frac / (val_frac + test_frac), stratify=rest["label"], random_state=seed
    )
    split_of = {
        **dict.fromkeys(train["lesion_id"], "train"),
        **dict.fromkeys(val["lesion_id"], "val"),
        **dict.fromkeys(test["lesion_id"], "test"),
    }
    out = meta[["image_id", "lesion_id", "dx", "label"]].copy()
    out["split"] = out["lesion_id"].map(split_of)
    return out.sort_values("image_id").reset_index(drop=True)


def check_split(split):
    """No lesion in two splits; print size and melanoma share per split."""
    per_lesion = split.groupby("lesion_id")["split"].nunique()
    assert (per_lesion == 1).all(), "leak: a lesion appears in more than one split"
    summary = split.groupby("split").agg(
        images=("image_id", "size"), lesions=("lesion_id", "nunique"), melanoma=("label", "sum")
    )
    summary["melanoma_%"] = (100 * summary["melanoma"] / summary["images"]).round(1)
    print(summary.loc[["train", "val", "test"]])


def build_cache(meta, image_ids, size, cache_file):
    """Resize every photo to size x size once, store as a uint8 array (N, size, size, 3)."""
    path_of = dict(zip(meta["image_id"], meta["path"]))
    images = np.empty((len(image_ids), size, size, 3), dtype=np.uint8)
    for i, image_id in enumerate(tqdm(image_ids, desc="resizing")):
        with Image.open(path_of[image_id]) as img:
            images[i] = np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, images)


def make_transform(cfg, train):
    """uint8 CHW tensor -> normalized float tensor; random augmentation on train only."""
    steps = []
    if train:
        aug = cfg["augment"]
        if aug["hflip"]:
            steps.append(v2.RandomHorizontalFlip())
        if aug["vflip"]:
            steps.append(v2.RandomVerticalFlip())
        steps.append(v2.RandomRotation(aug["rotation"]))
        cj = aug["color_jitter"]
        steps.append(v2.ColorJitter(brightness=cj, contrast=cj, saturation=cj))
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(steps)


class SkinDataset(Dataset):
    """Photos of one split, served from the cached array. Returns (image, label, index in split)."""

    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = torch.from_numpy(self.images[i]).permute(2, 0, 1)
        return self.transform(image), self.labels[i], i


def load_split(cfg):
    """The saved split table plus the cached image array, rows aligned."""
    split = pd.read_csv(cfg["data"]["split_file"])
    images = np.load(cfg["data"]["cache_file"])
    assert len(images) == len(split), "cache does not match split; rerun src/data.py"
    return split, images


def get_loaders(cfg, splits=("train", "val", "test")):
    """DataLoaders per split. Train is shuffled and augmented; val / test keep split.csv order."""
    split, images = load_split(cfg)
    loaders = {}
    for name in splits:
        rows = np.flatnonzero(split["split"] == name)
        is_train = name == "train"
        dataset = SkinDataset(images[rows], split["label"].values[rows], make_transform(cfg, is_train))
        loaders[name] = DataLoader(
            dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=is_train,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            persistent_workers=cfg["data"]["num_workers"] > 0,
        )
    return loaders


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/base.yaml")
    cfg = yaml.safe_load(open(parser.parse_args().config))
    d = cfg["data"]

    meta = load_metadata(d["root"])
    split = make_split(meta, d["split"], d["split_seed"])
    check_split(split)
    Path(d["split_file"]).parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(d["split_file"], index=False)
    print(f"saved {d['split_file']}")

    build_cache(meta, split["image_id"], d["image_size"], Path(d["cache_file"]))
    print(f"saved {d['cache_file']}")


if __name__ == "__main__":
    main()
