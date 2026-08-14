#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Materialize immutable RELiA-YOLO26 A0-A7 model YAMLs inside one run root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.utils import YAML

CONFIG_FILENAMES = {
    "A0": "relia_A0_baseline.yaml",
    "A1": "relia_A1_redpa.yaml",
    "A2": "relia_A2_psr.yaml",
    "A3": "relia_A3_dsq.yaml",
    "A4": "relia_A4_redpa_psr.yaml",
    "A5": "relia_A5_redpa_dsq.yaml",
    "A6": "relia_A6_psr_dsq.yaml",
    "A7": "relia_A7_full.yaml",
}


def materialize(identifier: str, output: Path) -> Path:
    """Write one audited, named A0-A7 configuration into an immutable run root."""
    if identifier not in CONFIG_FILENAMES:
        raise ValueError(f"RELiA experiment must be A0-A7, got {identifier}.")
    source = ROOT / "ultralytics/cfg/models/26" / CONFIG_FILENAMES[identifier]
    config = YAML.load(source)
    config.pop("yaml_file", None)
    config["scale"] = "n"
    relia = config.get("relia", {})
    enabled = bool(relia.get("enabled", False)) if isinstance(relia, dict) else bool(relia)
    if enabled != (identifier != "A0"):
        raise ValueError(f"{source} has an invalid relia.enabled value for {identifier}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(output, config, header="# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license\n")
    return output.resolve()


def main() -> None:
    """Materialize one requested RELiA ablation config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(CONFIG_FILENAMES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(materialize(args.experiment, args.output))


if __name__ == "__main__":
    main()
