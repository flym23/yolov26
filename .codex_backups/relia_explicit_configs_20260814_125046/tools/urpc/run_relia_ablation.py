#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run RELiA-YOLO26 A0-A7 with exactly three concurrent seeds per ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.urpc.materialize_relia_model import materialize  # noqa: E402


EXPERIMENTS, SEEDS = tuple(f"A{index}" for index in range(8)), (0, 1, 2)
ACTIVE: list[subprocess.Popen] = []
CANCELLED = False


def now() -> str:
    """Return a UTC timestamp suitable for immutable state artifacts."""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically update state/manifests without leaving a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sha256(path: Path) -> str:
    """Return a file-content SHA-256 for immutable manifests."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stop(processes: list[subprocess.Popen]) -> None:
    """Terminate the current three-worker process group on the first failure/cancellation."""
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and any(process.poll() is None for process in processes):
        time.sleep(0.2)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def signal_handler(_signum: int, _frame: Any) -> None:
    """Record cancellation and let the managed group exit together."""
    global CANCELLED
    CANCELLED = True


def job(args: argparse.Namespace, project: Path, identifier: str, seed: int) -> dict[str, Any]:
    """Create one fully absolute, reproducible worker specification."""
    model = materialize(identifier, project / "train" / "model_configs" / f"yolo26n-relia-{identifier.lower()}.yaml")
    training = {"epochs": 1 if args.stage == "smoke" else 300, "patience": 1 if args.stage == "smoke" else 40, "device": 0, "workers": 2, "amp": False, "deterministic": True, "plots": False, "imgsz": 640, "batch": 16}
    return {
        "experiment_id": identifier, "seed": seed, "stage": args.stage, "model": str(model), "model_sha256": sha256(model),
        "data": str(args.data), "weights": str(args.weights), "relia_stats": str(args.relia_stats), "training": training,
        "train_dir": str(project / "train" / identifier / f"seed{seed}"), "test_dir": str(project / "test" / identifier / f"seed{seed}"),
    }


def command(spec: dict[str, Any]) -> list[str]:
    """Use the project runtime and absolute worker path, preserving YOLO(...).load(...pt) in the worker."""
    return [sys.executable, str(ROOT / "tools/urpc/train_relia_yolo26_worker.py"), "--stage", spec["stage"], "--experiment", spec["experiment_id"], "--model", spec["model"], "--data", spec["data"], "--weights", spec["weights"], "--relia-stats", spec["relia_stats"], "--seed", str(spec["seed"]), "--train-dir", spec["train_dir"], "--test-dir", spec["test_dir"]]


def run_group(args: argparse.Namespace, project: Path, state: dict[str, Any], identifier: str) -> None:
    """Launch exactly seed 0/1/2, fail fast as a group, and persist PIDs/manifests."""
    global ACTIVE
    specs, handles, processes = [job(args, project, identifier, seed) for seed in SEEDS], [], []
    try:
        for spec in specs:
            log = project / "train" / "logs" / f"{identifier}_seed{spec['seed']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("x", encoding="utf-8")
            handles.append(handle)
            spec.update(status="running", started_at=now(), command=command(spec), log=str(log))
            atomic_json(project / "train" / "manifests" / f"{identifier}_seed{spec['seed']}.json", spec)
            processes.append(subprocess.Popen(spec["command"], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        if len(processes) != 3:
            raise RuntimeError("RELiA requires exactly three concurrent workers.")
        ACTIVE = processes
        state.update(worker_pids=[process.pid for process in processes], current_experiment=identifier, updated_at=now())
        atomic_json(project / "state.json", state)
        while True:
            if CANCELLED:
                raise KeyboardInterrupt("Cancellation requested")
            codes = [process.poll() for process in processes]
            failed = next(((spec, code) for spec, code in zip(specs, codes) if code not in (None, 0)), None)
            if failed:
                raise RuntimeError(f"{failed[0]['experiment_id']} seed={failed[0]['seed']} failed with exit code {failed[1]}; log={failed[0]['log']}")
            if all(code == 0 for code in codes):
                break
            time.sleep(2)
        for spec in specs:
            required = [Path(spec["train_dir"]) / "weights" / "last.pt", Path(spec["train_dir"]) / "results.csv"]
            if args.stage == "formal":
                required += [Path(spec["test_dir"]) / "summary_metrics.json", Path(spec["test_dir"]) / "scale_ap_metrics.json"]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError(f"Worker exited without required artifacts: {missing}")
            spec.update(status="completed", completed_at=now(), exit_code=0)
            atomic_json(project / "train" / "manifests" / f"{identifier}_seed{spec['seed']}.json", spec)
    except BaseException:
        stop(processes)
        for spec in specs:
            spec.update(status="cancelled" if CANCELLED else "failed", failed_at=now())
            atomic_json(project / "train" / "manifests" / f"{identifier}_seed{spec['seed']}.json", spec)
        raise
    finally:
        ACTIVE = []
        for handle in handles:
            handle.close()


def summarize(project: Path, identifiers: list[str]) -> None:
    """Write one per-ablation and one chain-level mean/std/best/worst test summary."""
    chain = []
    for identifier in identifiers:
        records = []
        for seed in SEEDS:
            record = json.loads((project / "test" / identifier / f"seed{seed}" / "summary_metrics.json").read_text(encoding="utf-8"))
            records.append({"seed": seed, **record["metrics"]})
        metrics = ("precision", "recall", "map50", "map50_95")
        aggregate = {metric: {"mean": statistics.mean(row[metric] for row in records), "std": statistics.stdev(row[metric] for row in records), "min": min(row[metric] for row in records), "max": max(row[metric] for row in records)} for metric in metrics}
        ranked = sorted(records, key=lambda row: row["map50_95"])
        summary = {"experiment_id": identifier, "seeds": records, "aggregate": aggregate, "best_seed": ranked[-1]["seed"], "worst_seed": ranked[0]["seed"], "generated_at": now()}
        atomic_json(project / "test" / identifier / "all_seeds_summary.json", summary)
        with (project / "test" / identifier / "all_seeds_summary.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("seed", *metrics)); writer.writeheader(); writer.writerows(records)
        chain.append(summary)
    atomic_json(project / "test" / "all_experiments_summary.json", {"experiments": chain, "generated_at": now()})


def main() -> None:
    """Run requested ablations serially, with each ablation's three seeds concurrent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--relia-stats", type=Path, required=True)
    parser.add_argument("--ids", default="A0,A1,A2,A3,A4,A5,A6,A7")
    parser.add_argument("--upstream-state", type=Path)
    parser.add_argument("--upstream-status", choices=("completed", "failed", "cancelled"))
    parser.add_argument("--upstream-reason", default="")
    args = parser.parse_args()
    project = args.project.resolve()
    if not project.is_absolute() or not all(path.is_file() for path in (args.data, args.weights, args.relia_stats)):
        raise ValueError("Project must be absolute and data/weights/relia-stats must exist.")
    identifiers = [identifier.strip().upper() for identifier in args.ids.split(",") if identifier.strip()]
    if not identifiers or any(identifier not in EXPERIMENTS for identifier in identifiers):
        raise ValueError("Only A0-A7 may be requested.")
    state_path = project / "state.json"
    existing = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    if existing and (existing.get("status") != "waiting" or existing.get("run_id") != args.run_id):
        raise FileExistsError(f"Run root is immutable: {state_path}")
    project.mkdir(parents=True, exist_ok=True)
    state = existing or {"schema_version": 1, "scheme": "relia_yolo26", "run_id": args.run_id, "stage": args.stage, "started_at": now()}
    state.update(status="running", phase="running", source_root=str(ROOT), run_root=str(project), data=str(args.data), weights=str(args.weights), relia_stats=str(args.relia_stats), experiments=identifiers, completed_experiments=[], worker_pids=[], updated_at=now())
    if args.upstream_state:
        state.update(upstream_state=str(args.upstream_state), upstream_status=args.upstream_status, upstream_failure_reason=args.upstream_reason)
    atomic_json(state_path, state)
    signal.signal(signal.SIGINT, signal_handler); signal.signal(signal.SIGTERM, signal_handler)
    try:
        for identifier in identifiers:
            run_group(args, project, state, identifier)
            state["completed_experiments"].append(identifier); state.update(worker_pids=[], updated_at=now())
            atomic_json(state_path, state)
        if args.stage == "formal":
            summarize(project, identifiers)
        state.update(status="completed", completed_at=now(), current_experiment=None, worker_pids=[])
        atomic_json(state_path, state)
    except BaseException as error:
        stop(ACTIVE)
        state.update(status="cancelled" if CANCELLED or isinstance(error, KeyboardInterrupt) else "failed", failed_at=now(), failure_reason=str(error), traceback=traceback.format_exc(), worker_pids=[])
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
