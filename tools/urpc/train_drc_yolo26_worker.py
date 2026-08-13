#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train and evaluate one DRC-YOLO26 ablation seed from any working directory."""

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


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically write one worker artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_paths(args: argparse.Namespace) -> None:
    """Reject non-local imports and incomplete inputs before allocating CUDA memory."""
    imported = Path(ultralytics.__file__).resolve()
    if ROOT not in imported.parents:
        raise RuntimeError(f"Imported ultralytics outside project root: {imported}")
    for label, path in (("dataset YAML", args.data), ("YOLO26-N weights", args.weights), ("model YAML", args.model)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    for path in (args.train_dir, args.test_dir):
        if not path.is_absolute():
            raise ValueError(f"Output paths must be absolute: {path}")
    if args.train_dir.exists() or (args.stage == "formal" and args.test_dir.exists()):
        raise FileExistsError(f"Refusing to overwrite an existing seed directory: {args.train_dir} or {args.test_dir}")


def metric_value(value: Any) -> float:
    """Convert a scalar tensor or NumPy value to a JSON-safe float."""
    return float(value.item()) if hasattr(value, "item") else float(value)


def evaluate_best(args: argparse.Namespace) -> None:
    """Evaluate the best checkpoint on the dataset test split and persist compact metrics."""
    best = args.train_dir / "weights" / "best.pt"
    if not best.is_file():
        best = args.train_dir / "weights" / "last.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce a checkpoint under {args.train_dir / 'weights'}")
    metrics = YOLO(best).val(
        data=str(args.data),
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        amp=False,
        plots=False,
        project=str(args.test_dir.parent),
        name=args.test_dir.name,
        exist_ok=False,
    )
    summary = {
        "experiment_id": args.experiment,
        "seed": args.seed,
        "checkpoint": str(best.resolve()),
        "metrics": {
            "precision": metric_value(metrics.box.mp),
            "recall": metric_value(metrics.box.mr),
            "map50": metric_value(metrics.box.map50),
            "map50_95": metric_value(metrics.box.map),
        },
        "speed_ms": {key: metric_value(value) for key, value in metrics.speed.items()},
    }
    atomic_json(args.test_dir / "summary_metrics.json", summary)
    atomic_json(
        args.test_dir / "scale_ap_metrics.json",
        {
            "status": "unavailable",
            "reason": "URPC2019 YOLO labels do not provide COCO object-area annotations for APs/APm/APl.",
        },
    )


def main() -> None:
    """Run one seed with the fixed user-requested training protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    validate_paths(args)

    model = YOLO(args.model).load(args.weights)
    model.train(
        data=str(args.data),
        imgsz=640,
        batch=16,
        epochs=1 if args.stage == "smoke" else 300,
        patience=1 if args.stage == "smoke" else 40,
        device=0,
        workers=2,
        amp=False,
        deterministic=True,
        plots=False,
        seed=args.seed,
        project=str(args.train_dir.parent),
        name=args.train_dir.name,
        exist_ok=False,
    )
    if not (args.train_dir / "weights" / "last.pt").is_file():
        raise RuntimeError(f"Training returned without {args.train_dir / 'weights' / 'last.pt'}")
    if args.stage == "formal":
        evaluate_best(args)


if __name__ == "__main__":
    main()
