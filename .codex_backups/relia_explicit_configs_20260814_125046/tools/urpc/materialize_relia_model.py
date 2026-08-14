#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Materialize immutable RELiA-YOLO26 A0-A7 model YAMLs inside one run root."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML  # noqa: E402


FLAGS = {
    "A1": {"redpa": True, "psr": False, "dsq": False},
    "A2": {"redpa": False, "psr": True, "dsq": False},
    "A3": {"redpa": False, "psr": False, "dsq": True},
    "A4": {"redpa": True, "psr": True, "dsq": False},
    "A5": {"redpa": True, "psr": False, "dsq": True},
    "A6": {"redpa": False, "psr": True, "dsq": True},
    "A7": {"redpa": True, "psr": True, "dsq": True},
}


def materialize(identifier: str, output: Path) -> Path:
    """Write A0 as exact YOLO26 and A1-A7 as explicit RELiA feature combinations."""
    if identifier != "A0" and identifier not in FLAGS:
        raise ValueError(f"RELiA experiment must be A0-A7, got {identifier}.")
    source = ROOT / "ultralytics/cfg/models/26" / ("yolo26.yaml" if identifier == "A0" else "yolo26-relia.yaml")
    config = deepcopy(YAML.load(source))
    config.pop("yaml_file", None)
    config["scale"] = "n"
    if identifier != "A0":
        config["experiment_id"] = identifier
        config["relia"].update(FLAGS[identifier])
    output.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(output, config, header="# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license\n")
    return output.resolve()


def main() -> None:
    """Materialize one requested RELiA ablation config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A0", *FLAGS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.experiment, args.output))


if __name__ == "__main__":
    main()
