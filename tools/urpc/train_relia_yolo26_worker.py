#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train and evaluate one RELiA-YOLO26 ablation seed from any working directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a compact worker artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def metric_value(value: Any) -> float:
    """Return scalar tensor/NumPy metrics in JSON-compatible form."""
    return float(value.item()) if hasattr(value, "item") else float(value)


def validate(args: argparse.Namespace) -> None:
    """Reject incomplete paths or accidental site-packages imports before CUDA allocation."""
    imported = Path(ultralytics.__file__).resolve()
    if ROOT not in imported.parents:
        raise RuntimeError(f"Imported ultralytics outside project root: {imported}")
    required = (("data", args.data), ("model", args.model), ("weights", args.weights))
    if args.experiment != "A0":
        required += (("RELiA statistics", args.relia_stats),)
    for label, path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not args.train_dir.is_absolute() or not args.test_dir.is_absolute():
        raise ValueError("RELiA output paths must be absolute.")
    if args.train_dir.exists() or (args.stage == "formal" and args.test_dir.exists()):
        raise FileExistsError(f"Refusing to overwrite seed output: {args.train_dir} or {args.test_dir}")


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate the best model on URPC2019 test and persist per-seed metrics."""
    checkpoint = args.train_dir / "weights" / "best.pt"
    checkpoint = checkpoint if checkpoint.is_file() else args.train_dir / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No checkpoint under {args.train_dir / 'weights'}")
    metrics = YOLO(checkpoint).val(
        data=str(args.data), split="test", imgsz=640, batch=16, device=0, workers=2, amp=False, plots=False,
        project=str(args.test_dir.parent), name=args.test_dir.name, exist_ok=False,
    )
    atomic_json(args.test_dir / "summary_metrics.json", {
        "experiment_id": args.experiment, "seed": args.seed, "checkpoint": str(checkpoint.resolve()),
        "metrics": {"precision": metric_value(metrics.box.mp), "recall": metric_value(metrics.box.mr), "map50": metric_value(metrics.box.map50), "map50_95": metric_value(metrics.box.map)},
        "speed_ms": {key: metric_value(value) for key, value in metrics.speed.items()},
    })
    atomic_json(args.test_dir / "scale_ap_metrics.json", {"status": "unavailable", "reason": "URPC2019 YOLO labels have no canonical object-area AP split."})


def main() -> None:
    """Run one requested seed with the immutable training protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--experiment", required=True, choices=tuple(f"A{index}" for index in range(8)))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--relia-stats", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    args = parser.parse_args()
    validate(args)
    model = YOLO(args.model).load(args.weights)
    options = {
        "data": str(args.data), "imgsz": 640, "batch": 16, "epochs": 1 if args.stage == "smoke" else 300,
        "patience": 1 if args.stage == "smoke" else 40, "device": 0, "workers": 2, "amp": False,
        "deterministic": True, "plots": False, "seed": args.seed, "project": str(args.train_dir.parent),
        "name": args.train_dir.name, "exist_ok": False,
    }
    if args.experiment != "A0":
        options["relia_stats"] = str(args.relia_stats)
    model.train(**options)
    if not (args.train_dir / "weights" / "last.pt").is_file():
        raise RuntimeError(f"Training returned without {args.train_dir / 'weights' / 'last.pt'}")
    if args.stage == "formal":
        evaluate(args)


if __name__ == "__main__":
    main()
