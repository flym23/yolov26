#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Materialize immutable YAMLs for the DRC-YOLO26 ablation matrix.

The canonical module layouts remain the checked-in baseline/CDR/SDR/CCQ/DRC YAMLs.  This utility only selects
one of those layouts and writes its documented, declarative ablation switches to a run-local YAML.  The launcher
records the resulting SHA-256 in its immutable manifest, so a run never relies on a mutable in-memory architecture.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML  # noqa: E402


MODEL_DIR = ROOT / "ultralytics/cfg/models/26"
BASE_MODELS = {
    "baseline": "yolo26.yaml",
    "cdr": "yolo26-cdr.yaml",
    "sdr": "yolo26-sdr.yaml",
    "ccq": "yolo26-ccq.yaml",
    "drc": "yolo26-drc.yaml",
}
CDR_FLAGS = {
    "D1": {"use_opponent": True, "use_dual_highpass": False, "use_reliability": False, "use_router": False, "zero_init": False},
    "D2": {"use_opponent": True, "use_dual_highpass": True, "use_reliability": False, "use_router": False, "zero_init": False},
    "D3": {"use_opponent": True, "use_dual_highpass": True, "use_reliability": True, "use_router": False, "zero_init": False},
}
SDR_FLAGS = {
    "S1": {"mode": "stride_concat", "use_reliability_gate": False, "zero_init": False},
    "S2": {"mode": "pixel_concat", "use_reliability_gate": False, "zero_init": False},
    "S3": {"mode": "route", "use_reliability_gate": False, "zero_init": False},
    "S4": {"mode": "route", "use_reliability_gate": True, "zero_init": False},
}
CCQ_FLAGS = {
    "Q1": {"use_consensus": True, "use_conditional_gate": False, "use_reliability": False, "use_quality": False},
    "Q2": {"use_consensus": True, "use_conditional_gate": True, "use_reliability": False, "use_quality": False},
    "Q3": {"use_consensus": True, "use_conditional_gate": True, "use_reliability": True, "use_quality": False},
    "Q4": {"use_consensus": False, "use_conditional_gate": False, "use_reliability": False, "use_quality": True},
}
RECIPES = {
    "A4": ("drc", "detect"),
    "A5": ("cdr", "ccq"),
    "A6": ("sdr", "ccq"),
    **{key: ("cdr", "detect") for key in CDR_FLAGS},
    **{key: ("sdr", "detect") for key in SDR_FLAGS},
    **{key: ("ccq", "ccq") for key in CCQ_FLAGS},
}


def materialize(config_id: str, output: Path) -> Path:
    """Write one fixed ablation YAML and return its path."""
    if config_id not in RECIPES:
        raise KeyError(f"No materialized model recipe for {config_id}; use its canonical YAML or declared alias.")
    base_name, head_kind = RECIPES[config_id]
    data = deepcopy(YAML.load(MODEL_DIR / BASE_MODELS[base_name]))
    data.pop("yaml_file", None)
    data["scale"] = "n"
    data["experiment_id"] = config_id
    if config_id in CDR_FLAGS:
        data["backbone"][0][3].append(CDR_FLAGS[config_id])
    if config_id in SDR_FLAGS:
        data["head"][4][3].append(SDR_FLAGS[config_id])
    if head_kind == "detect":
        data["head"][-1][2] = "Detect"
        data["head"][-1][3] = ["nc"]
        data.pop("ccq", None)
    elif head_kind == "ccq":
        data["head"][-1][2] = "CCQDetect"
        data["head"][-1][3] = ["nc", 0.50]
        data.setdefault("ccq", {}).update(CCQ_FLAGS.get(config_id, {}))
        if config_id in CCQ_FLAGS and not data["ccq"]["use_quality"]:
            data["head"][-1][3].append(False)
    YAML.save(output, data, header="# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license\n")
    return output


def main() -> None:
    """Materialize a requested model YAML from its checked-in recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(RECIPES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.config, args.output).resolve())


if __name__ == "__main__":
    main()
