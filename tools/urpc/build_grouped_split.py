#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Create a leakage-resistant URPC split by grouping visually similar frames."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class UnionFind:
    """Small disjoint-set structure for near-duplicate image groups."""

    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, index: int) -> int:
        """Return the representative for an item."""
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        """Merge two components."""
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def phash(image: np.ndarray) -> int:
    """Return a 64-bit DCT perceptual hash without an external image-hash dependency."""
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(gray)[:8, :8]
    bits = low > np.median(low[1:, :])
    return int("".join("1" if value else "0" for value in bits.flat), 2)


def label_path(image: Path, images: Path, labels: Path) -> Path:
    """Map an image below the image root to its matching YOLO label path."""
    return (labels / image.relative_to(images)).with_suffix(".txt")


def label_counts(path: Path) -> Counter:
    """Read class counts from a YOLO label file."""
    counts = Counter()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 5:
                counts[int(fields[0])] += 1
    return counts


def ssim(left: np.ndarray, right: np.ndarray) -> float:
    """Compute a lightweight normalized similarity for pHash candidate confirmation."""
    left, right = left.astype(np.float32), right.astype(np.float32)
    left -= left.mean()
    right -= right.mean()
    return float((left * right).mean() / (left.std() * right.std() + 1e-6))


def main() -> None:
    """Build group-aware train, validation, and test image lists."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hamming", type=int, default=8)
    parser.add_argument("--ssim", type=float, default=0.92)
    args = parser.parse_args()
    images = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found under {args.images}")
    thumbnails, hashes, counts = [], [], []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")
        thumbnails.append(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA))
        hashes.append(phash(image))
        counts.append(label_counts(label_path(image_path, args.images, args.labels)))
    buckets = defaultdict(list)
    for index, value in enumerate(hashes):
        for shift in range(0, 64, 16):
            buckets[(shift, (value >> shift) & 0xFFFF)].append(index)
    groups = UnionFind(len(images))
    checked = set()
    for candidates in buckets.values():
        for offset, left in enumerate(candidates):
            for right in candidates[offset + 1 :]:
                pair = (left, right) if left < right else (right, left)
                if pair in checked:
                    continue
                checked.add(pair)
                if (hashes[left] ^ hashes[right]).bit_count() <= args.hamming and ssim(thumbnails[left], thumbnails[right]) >= args.ssim:
                    groups.union(left, right)
    components = defaultdict(list)
    for index in range(len(images)):
        components[groups.find(index)].append(index)
    grouped = list(components.values())
    random.Random(args.seed).shuffle(grouped)
    targets = {"train": 0.70 * len(images), "val": 0.10 * len(images), "test": 0.20 * len(images)}
    selected = {name: [] for name in targets}
    current = {name: 0 for name in targets}
    for group in sorted(grouped, key=len, reverse=True):
        split = min(targets, key=lambda name: (current[name] - targets[name], current[name]))
        selected[split].extend(group)
        current[split] += len(group)
    split_root = args.output / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    for name, indices in selected.items():
        (split_root / f"{name}.txt").write_text("\n".join(str(images[index].resolve()) for index in sorted(indices)) + "\n", encoding="utf-8")
    stats = {
        "seed": args.seed,
        "images": {name: len(indices) for name, indices in selected.items()},
        "instances": {name: sum(sum(counts[index].values()) for index in indices) for name, indices in selected.items()},
        "class_instances": {name: dict(sum((counts[index] for index in indices), Counter())) for name, indices in selected.items()},
        "group_sizes": sorted(len(group) for group in grouped),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "groups.json").write_text(json.dumps(grouped, indent=2), encoding="utf-8")
    (args.output / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    data = {
        "path": str(args.images.parent.resolve()),
        "train": str((split_root / "train.txt").resolve()),
        "val": str((split_root / "val.txt").resolve()),
        "test": str((split_root / "test.txt").resolve()),
        "names": ["holothurian", "echinus", "scallop", "starfish"],
    }
    with (args.output / "urpc2020half_grouped.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
