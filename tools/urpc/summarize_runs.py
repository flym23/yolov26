#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validate manifest-backed URPC runs and write publication-result table skeletons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml


MATRIX_PATH = Path(__file__).with_name("experiment_matrix.yaml")
SEEDS = {0, 1, 2}
REQUIRED_TRAINING = ("imgsz", "batch", "epochs", "patience", "device", "workers", "amp", "deterministic", "plots", "optimizer", "cos_lr", "close_mosaic", "data", "pretrained")


def atomic_csv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    """Write one result table atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_manifests(project: Path) -> list[dict[str, Any]]:
    """Load only complete or exact-reuse manifests from the scheduler."""
    records = []
    for path in sorted((project / "manifests").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("status") in {"complete", "reused"}:
            record["manifest_path"] = str(path)
            records.append(record)
    return records


def result_row(record: dict[str, Any]) -> dict[str, str]:
    """Read the final Ultralytics results row, retaining an empty skeleton when unavailable."""
    results = Path(record["artifact_dir"]) / "results.csv"
    if not results.is_file():
        return {}
    with results.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return rows[-1] if rows else {}


def metric(row: dict[str, str], *names: str) -> float | None:
    """Return a metric from either current or historical Ultralytics CSV header names."""
    for name in names:
        try:
            return float(row[name])
        except (KeyError, TypeError, ValueError):
            pass
    return None


def signature(record: dict[str, Any], include_pretrained: bool) -> str:
    """Return the comparable configuration signature, optionally including the base weights."""
    keys = REQUIRED_TRAINING if include_pretrained else tuple(key for key in REQUIRED_TRAINING if key != "pretrained")
    payload = {key: record.get("training", {}).get(key) for key in keys} | {"dataset": record.get("dataset")}
    return json.dumps(payload, sort_keys=True)


def main() -> None:
    """Validate all required seed runs and emit the section-23 result artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True, help="The run_ablation.py project directory.")
    parser.add_argument("--artifacts", type=Path, help="Output directory; defaults to <project>/artifacts.")
    parser.add_argument("--ids", help="Expected comma-separated IDs; defaults to the entire C/A/D/S/Q matrix.")
    args = parser.parse_args()
    project = args.project.resolve()
    artifacts = (args.artifacts or project / "artifacts").resolve()
    matrix = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    expected = tuple(item.strip() for item in args.ids.split(",")) if args.ids else tuple(matrix["experiments"])
    records = load_manifests(project)
    by_id: dict[str, list[dict[str, Any]]] = {identifier: [] for identifier in expected}
    for record in records:
        if record.get("experiment_id") in by_id:
            by_id[record["experiment_id"]].append(record)
    invalid: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    global_signatures: set[str] = set()
    for identifier, group in by_id.items():
        seeds = {record.get("seed") for record in group}
        if seeds != SEEDS:
            invalid.append({"experiment_id": identifier, "reason": f"missing or duplicate seeds: found {sorted(seeds)}"})
            continue
        signatures = {signature(record, include_pretrained=True) for record in group}
        if len(signatures) != 1:
            invalid.append({"experiment_id": identifier, "reason": "seed runs have incompatible training/data fingerprints"})
            continue
        global_signatures.update(signature(record, include_pretrained=False) for record in group)
        if any(not (Path(record["artifact_dir"]) / "weights" / "last.pt").is_file() for record in group):
            invalid.append({"experiment_id": identifier, "reason": "canonical artifact is missing weights/last.pt"})
            continue
        accepted.extend(group)
    if len(global_signatures) > 1:
        invalid.append({"experiment_id": "ALL", "reason": "models do not share one unified training/data configuration"})
        accepted = []
    pretrained_weights = {record["training"]["pretrained"] for record in accepted}
    if len(pretrained_weights) > 1:
        invalid.append({"experiment_id": "ALL", "reason": "models use different pretrained weights"})
        accepted = []
    main_rows: list[dict[str, Any]] = []
    for identifier in expected:
        group = [record for record in accepted if record["experiment_id"] == identifier]
        if not group:
            continue
        rows = [result_row(record) for record in group]
        values = [metric(row, "metrics/mAP50-95(B)", "metrics/mAP50-95") for row in rows]
        values = [value for value in values if value is not None]
        main_rows.append(
            {
                "experiment_id": identifier,
                "seeds": ";".join(str(record["seed"]) for record in sorted(group, key=lambda item: item["seed"])),
                "precision_mean": _mean([metric(row, "metrics/precision(B)", "metrics/precision") for row in rows]),
                "recall_mean": _mean([metric(row, "metrics/recall(B)", "metrics/recall") for row in rows]),
                "map50_mean": _mean([metric(row, "metrics/mAP50(B)", "metrics/mAP50") for row in rows]),
                "map50_95_mean": _mean(values),
                "map50_95_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "model_sha256": group[0]["model"]["sha256"],
                "manifest": group[0]["manifest_path"],
            }
        )
    atomic_csv(artifacts / "urpc2020half_main_results.csv", list(main_rows[0]) if main_rows else ["experiment_id", "seeds", "precision_mean", "recall_mean", "map50_mean", "map50_95_mean", "map50_95_std", "model_sha256", "manifest"], main_rows)
    atomic_csv(
        artifacts / "urpc2020half_ablation_results.csv",
        ["experiment_id", "status", "reason"],
        [{"experiment_id": row["experiment_id"], "status": "valid", "reason": ""} for row in main_rows]
        + [{**row, "status": "invalid"} for row in invalid],
    )
    atomic_csv(artifacts / "urpc2020half_per_class_results.csv", ["experiment_id", "class", "ap50", "ap50_95"], [])
    atomic_csv(artifacts / "urpc2020half_complexity_results.csv", ["experiment_id", "params", "gflops", "peak_gpu_memory", "pytorch_fp32_latency", "pytorch_fp16_latency", "onnxruntime_latency", "tensorrt_fp16_latency"], [])
    bootstrap = {"status": "invalid" if invalid else "pending_per_image_predictions", "resamples": 1000, "confidence_interval": 0.95, "comparisons": [{"baseline": "A0", "candidate": "A7", "metrics": ["map50_95", "map75", "aps", "recall"]}], "invalid_runs": invalid}
    (artifacts / "urpc2020half_bootstrap_results.json").write_text(json.dumps(bootstrap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if invalid:
        raise SystemExit(f"Rejected {len(invalid)} invalid or incomplete experiment groups; see {artifacts}.")


def _mean(values: list[float | None]) -> float:
    """Return a NaN-safe mean for optional CSV metrics."""
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.mean(finite) if finite else math.nan


if __name__ == "__main__":
    main()
