#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate COCO detection predictions and export AP/AP50/AP75/AP_s/AP_m/AP_l."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(annotations: Path, predictions: Path) -> dict[str, float]:
    """Return standard COCO bounding-box AP metrics for a prediction JSON file."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError("eval_size_ap.py requires pycocotools; install it in the experiment environment.") from error
    coco_gt = COCO(str(annotations))
    coco_dt = coco_gt.loadRes(str(predictions))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ("AP", "AP50", "AP75", "APs", "APm", "APl")
    return {name: float(evaluator.stats[index]) for index, name in enumerate(names)}


def main() -> None:
    """Run COCO size-stratified evaluation and write a compact machine-readable result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Ground-truth COCO annotations JSON.")
    parser.add_argument("--predictions", type=Path, required=True, help="COCO detection-results JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output metrics JSON.")
    args = parser.parse_args()
    for path in (args.annotations, args.predictions):
        if not path.is_file():
            raise FileNotFoundError(path)
    metrics = evaluate(args.annotations, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
