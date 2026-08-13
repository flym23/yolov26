#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Run the DRC-YOLO26 A0-A7 chain with three concurrent seeds per ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.urpc.materialize_experiment_model import RECIPES, materialize  # noqa: E402


MATRIX_PATH = Path(__file__).with_name("experiment_matrix.yaml")
DEFAULT_PROJECT = ROOT / "runs" / "drc_yolo26_a0_a7"
EXPERIMENTS = tuple(f"A{index}" for index in range(8))
SEEDS = (0, 1, 2)
TRAIN_KEYS = ("imgsz", "batch", "epochs", "patience", "device", "workers", "amp", "deterministic", "plots")
CODE_FILES = (
    "ultralytics/nn/modules/block.py",
    "ultralytics/nn/modules/head.py",
    "ultralytics/nn/tasks.py",
    "ultralytics/utils/loss.py",
)
ACTIVE_PROCESSES: list[subprocess.Popen] = []
CANCEL_REQUESTED = False


def utcnow() -> str:
    """Return an unambiguous UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


@lru_cache
def sha256_file(path: Path) -> str:
    """Return a content hash for one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    """Return a SHA-256 hash for JSON-compatible input."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically create or replace one JSON document."""
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


def model_path(value: str) -> Path:
    """Resolve a model YAML, including scale aliases such as yolo26n-drc.yaml."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if candidate.is_file():
        return candidate.resolve()
    name = candidate.name
    for scale in "nsm lx".replace(" ", ""):
        marker = f"yolo26{scale}"
        if name.startswith(marker):
            resolved = candidate.with_name(f"yolo26{name[len(marker):]}")
            if resolved.is_file():
                return resolved.resolve()
    raise FileNotFoundError(f"Model YAML does not exist and no scale alias resolved it: {value}")


@lru_cache
def dataset_fingerprint(data: Path) -> dict[str, str]:
    """Fingerprint the dataset YAML and its explicit split inventories."""
    payload = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    root = Path(payload.get("path", data.parent)).expanduser()
    if not root.is_absolute():
        root = (data.parent / root).resolve()
    hashes = {"data_yaml": sha256_file(data)}
    for split in ("train", "val", "test"):
        value = payload.get(split)
        if not isinstance(value, str):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            hashes[f"{split}_list"] = sha256_file(path)
        elif path.is_dir():
            inventory = [
                (str(item.relative_to(path)), item.stat().st_size)
                for item in sorted(path.rglob("*"))
                if item.is_file()
            ]
            hashes[f"{split}_inventory"] = stable_hash(inventory)
        else:
            raise FileNotFoundError(f"Dataset {split!r} does not exist: {path}")
    labels = root / "labels"
    if labels.is_dir():
        digest = hashlib.sha256()
        for path in sorted(labels.rglob("*.txt")):
            digest.update(str(path.relative_to(labels)).encode())
            digest.update(path.read_bytes())
        hashes["labels"] = digest.hexdigest()
    return hashes


def environment() -> dict[str, Any]:
    """Capture host facts needed to interpret a run."""
    details: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": None,
        "cuda": None,
        "gpu": None,
    }
    try:
        import torch

        details.update(torch=torch.__version__, cuda=torch.version.cuda)
        if torch.cuda.is_available():
            details["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        details["ultralytics"] = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        details["ultralytics"] = "local-source"
    return details


def git_commit() -> str | None:
    """Return the checked-out commit when the workspace is a Git checkout."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def default_config() -> dict[str, Any]:
    """Return the fixed DRC-YOLO26 A0-A7 training configuration."""
    matrix = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
    return {
        "data": None,
        "pretrained": "yolo26n.pt",
        "project": str(DEFAULT_PROJECT),
        "imgsz": 640,
        "batch": 16,
        "epochs": 300,
        "patience": 40,
        "device": 0,
        "workers": 2,
        "amp": False,
        "deterministic": True,
        "plots": False,
        "seeds": list(SEEDS),
        "models": matrix["models"],
        "experiments": matrix["experiments"],
    }


def merge_config(config: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge the documented config surface while preserving the model matrix."""
    merged = dict(config)
    for key, value in override.items():
        merged[key] = {**merged[key], **(value or {})} if key in {"models", "experiments"} else value
    return merged


def load_config(path: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    """Load an optional YAML over the fixed experiment defaults."""
    config = default_config()
    if path:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("The scheduler configuration must be a YAML mapping.")
        config = merge_config(config, loaded)
    for key in ("data", "pretrained", "project"):
        value = getattr(args, key, None)
        if value:
            config[key] = value
    if not config.get("data"):
        raise ValueError("Set data in the YAML configuration or pass --data.")
    required = {
        "imgsz": 640,
        "batch": 16,
        "epochs": 300,
        "patience": 40,
        "device": 0,
        "workers": 2,
        "amp": False,
        "deterministic": True,
        "plots": False,
    }
    mismatched = {key: config.get(key) for key, expected in required.items() if config.get(key) != expected}
    if mismatched:
        raise ValueError(f"Training configuration differs from the fixed DRC-YOLO26 protocol: {mismatched}")
    if list(config["seeds"]) != list(SEEDS):
        raise ValueError("The DRC-YOLO26 protocol requires exactly seeds [0, 1, 2].")
    return config


def selected_experiments(config: dict[str, Any], requested: str | None) -> list[str]:
    """Return requested A0-A7 identifiers in canonical order."""
    wanted = list(EXPERIMENTS) if not requested else [item.strip().upper() for item in requested.split(",") if item.strip()]
    unknown = sorted(set(wanted) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Only A0-A7 are valid for this chain; received: {', '.join(unknown)}")
    missing = sorted(set(wanted) - set(config["experiments"]))
    if missing:
        raise ValueError(f"Experiment matrix is missing: {', '.join(missing)}")
    return [identifier for identifier in EXPERIMENTS if identifier in wanted]


def resolve_experiment_model(experiment_id: str, declared: str, project: Path) -> tuple[Path, str]:
    """Resolve a canonical YAML or materialize one documented combination."""
    try:
        resolved = model_path(declared)
    except FileNotFoundError:
        if experiment_id not in RECIPES:
            raise
        output = project / "model_configs" / f"yolo26n-{experiment_id.lower()}.yaml"
        if not output.exists():
            materialize(experiment_id, output)
        resolved = output.resolve()
    return resolved, sha256_file(resolved)


def job_spec(config: dict[str, Any], experiment_id: str, seed: int, project: Path, stage: str) -> dict[str, Any]:
    """Create one immutable seed specification."""
    experiment = config["experiments"][experiment_id]
    if isinstance(experiment, str):
        experiment = {"model": experiment}
    model_key = experiment["model"]
    model = config["models"][model_key]
    if isinstance(model, str):
        model = {"yaml": model}
    declared = str(model["yaml"])
    data = Path(config["data"]).expanduser().resolve()
    pretrained = Path(model.get("pretrained", config["pretrained"])).expanduser()
    if not pretrained.is_absolute():
        pretrained = ROOT / pretrained
    pretrained = pretrained.resolve()
    resolved_model, model_sha256 = resolve_experiment_model(experiment_id, declared, project)
    if not data.is_file() or not pretrained.is_file():
        raise FileNotFoundError(f"Missing data or weights: data={data}, pretrained={pretrained}")
    training = {key: config[key] for key in TRAIN_KEYS} | {"data": str(data), "pretrained": str(pretrained)}
    train_dir = project / "train" / experiment_id / f"seed{seed}"
    test_dir = project / "test" / experiment_id / f"seed{seed}"
    fingerprint_input = {
        "model": {"declared": declared, "resolved": str(resolved_model), "sha256": model_sha256},
        "training": training,
        "seed": seed,
        "dataset": dataset_fingerprint(data),
        "pretrained_sha256": sha256_file(pretrained),
        "code": {path: sha256_file(ROOT / path) for path in CODE_FILES},
    }
    return {
        "schema_version": 2,
        "stage": stage,
        "experiment_id": experiment_id,
        "experiment": experiment,
        "seed": seed,
        "fingerprint": stable_hash(fingerprint_input),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "model": fingerprint_input["model"] | {"key": model_key},
        "training": training,
        "dataset": fingerprint_input["dataset"],
        "pretrained_sha256": fingerprint_input["pretrained_sha256"],
        "code": fingerprint_input["code"],
        "git_commit": git_commit(),
        "environment": environment(),
        "created_at": utcnow(),
    }


def command_for(spec: dict[str, Any]) -> list[str]:
    """Build one project-root-safe worker command."""
    return [
        sys.executable,
        str(ROOT / "tools/urpc/train_drc_yolo26_worker.py"),
        "--stage",
        spec["stage"],
        "--experiment",
        spec["experiment_id"],
        "--model",
        spec["model"]["resolved"],
        "--data",
        spec["training"]["data"],
        "--weights",
        spec["training"]["pretrained"],
        "--seed",
        str(spec["seed"]),
        "--train-dir",
        spec["train_dir"],
        "--test-dir",
        spec["test_dir"],
    ]


def stop_processes(processes: list[subprocess.Popen]) -> None:
    """Terminate a worker batch, including dataloader descendants."""
    alive = [process for process in processes if process.poll() is None]
    for process in alive:
        try:
            os.killpg(process.pid, signal.SIGTERM) if os.name == "posix" else process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    while alive and time.monotonic() < deadline:
        alive = [process for process in alive if process.poll() is None]
        time.sleep(0.2)
    for process in alive:
        try:
            os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        except ProcessLookupError:
            pass
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def handle_signal(_signum: int, _frame: Any) -> None:
    """Request orderly cancellation from SIGINT or SIGTERM."""
    global CANCEL_REQUESTED
    CANCEL_REQUESTED = True


def run_group(jobs: list[dict[str, Any]], state_path: Path, state: dict[str, Any]) -> None:
    """Run exactly three seed workers concurrently and fail the batch together."""
    global ACTIVE_PROCESSES
    log_handles = []
    processes: list[subprocess.Popen] = []
    try:
        for job in jobs:
            log_path = state_path.parent / "train" / "logs" / f"{job['experiment_id']}_seed{job['seed']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("x", encoding="utf-8")
            log_handles.append(handle)
            command = command_for(job)
            job.update(status="running", started_at=utcnow(), command=command, log=str(log_path))
            atomic_json(state_path.parent / "train" / "manifests" / f"{job['experiment_id']}_seed{job['seed']}.json", job)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        if len(processes) != 3:
            raise RuntimeError(f"Expected three concurrent workers, launched {len(processes)}.")
        ACTIVE_PROCESSES = processes
        state.update(worker_pids=[process.pid for process in processes], updated_at=utcnow())
        atomic_json(state_path, state)
        while True:
            if CANCEL_REQUESTED:
                raise KeyboardInterrupt("Cancellation requested")
            return_codes = [process.poll() for process in processes]
            failed = [(jobs[index], code) for index, code in enumerate(return_codes) if code not in (None, 0)]
            if failed:
                job, code = failed[0]
                raise RuntimeError(
                    f"{job['experiment_id']} seed={job['seed']} failed with exit code {code}; see {job['log']}"
                )
            if all(code == 0 for code in return_codes):
                break
            time.sleep(2)
        for job in jobs:
            train_dir = Path(job["train_dir"])
            required = [train_dir / "weights" / "last.pt", train_dir / "results.csv"]
            if job["stage"] == "formal":
                required.extend((Path(job["test_dir"]) / name for name in ("summary_metrics.json", "scale_ap_metrics.json")))
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError(f"Worker completed without required artifacts: {missing}")
            job.update(status="completed", completed_at=utcnow(), exit_code=0)
            atomic_json(state_path.parent / "train" / "manifests" / f"{job['experiment_id']}_seed{job['seed']}.json", job)
    except BaseException:
        stop_processes(processes)
        status = "cancelled" if CANCEL_REQUESTED else "failed"
        for job in jobs:
            job.update(status=status, failed_at=utcnow())
            atomic_json(
                state_path.parent / "train" / "manifests" / f"{job['experiment_id']}_seed{job['seed']}.json",
                job,
            )
        raise
    finally:
        ACTIVE_PROCESSES = []
        for handle in log_handles:
            handle.close()


def summarize_experiment(project: Path, identifier: str) -> dict[str, Any]:
    """Write JSON and CSV summaries for one completed three-seed experiment."""
    records = []
    for seed in SEEDS:
        path = project / "test" / identifier / f"seed{seed}" / "summary_metrics.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        records.append({"seed": seed, **record["metrics"]})
    metrics = ("precision", "recall", "map50", "map50_95")
    aggregate = {}
    for metric in metrics:
        values = [float(record[metric]) for record in records]
        aggregate[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }
    ranked = sorted(records, key=lambda record: record["map50_95"])
    summary = {
        "experiment_id": identifier,
        "seeds": records,
        "aggregate": aggregate,
        "best_seed": ranked[-1]["seed"],
        "worst_seed": ranked[0]["seed"],
        "generated_at": utcnow(),
    }
    output = project / "test" / identifier
    atomic_json(output / "all_seeds_summary.json", summary)
    with (output / "all_seeds_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("seed", *metrics))
        writer.writeheader()
        writer.writerows(records)
    return summary


def summarize_chain(project: Path, summaries: list[dict[str, Any]]) -> None:
    """Write the final cross-experiment summary artifacts."""
    atomic_json(project / "test" / "all_experiments_summary.json", {"experiments": summaries, "generated_at": utcnow()})
    fields = ["experiment_id", "best_seed", "worst_seed"]
    for metric in ("precision", "recall", "map50", "map50_95"):
        fields.extend(f"{metric}_{name}" for name in ("mean", "std", "min", "max"))
    rows = []
    for summary in summaries:
        row = {key: summary[key] for key in ("experiment_id", "best_seed", "worst_seed")}
        for metric, values in summary["aggregate"].items():
            row.update({f"{metric}_{name}": value for name, value in values.items()})
        rows.append(row)
    with (project / "test" / "all_experiments_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Execute A0-A7 sequentially, with seeds 0/1/2 concurrent within each experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--project", required=True, help="Absolute immutable run root.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--ids", help="Comma-separated subset of A0-A7; defaults to all eight.")
    parser.add_argument("--upstream-state", help="Recorded terminal state that released this chain.")
    parser.add_argument("--upstream-status", choices=("completed", "failed", "cancelled"))
    parser.add_argument("--upstream-reason", default="")
    args = parser.parse_args()
    project = Path(args.project).expanduser()
    if not project.is_absolute():
        raise ValueError(f"--project must be absolute: {project}")
    project = project.resolve()
    state_path = project / "state.json"
    waiting_state = None
    if state_path.exists():
        try:
            waiting_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Existing run state is unreadable: {state_path}") from error
        if (
            waiting_state.get("status") != "waiting"
            or waiting_state.get("run_id") != args.run_id
            or not args.upstream_state
        ):
            raise FileExistsError(f"Run ID is not immutable/unique because state already exists: {state_path}")
    config = load_config(args.config, args)
    identifiers = selected_experiments(config, args.ids)
    project.mkdir(parents=True, exist_ok=True)
    state = waiting_state or {
        "schema_version": 1,
        "scheme": "drc_yolo26_a0_a7",
        "run_id": args.run_id,
        "stage": args.stage,
        "started_at": utcnow(),
    }
    state.update(
        status="running",
        phase="running",
        source_root=str(ROOT),
        run_root=str(project),
        data=str(Path(args.data).resolve()),
        weights=str(Path(args.pretrained).resolve()),
        experiments=identifiers,
        completed_experiments=[],
        worker_pids=[],
        updated_at=utcnow(),
    )
    if args.upstream_state:
        state.update(
            upstream_state=str(Path(args.upstream_state).resolve()),
            upstream_status=args.upstream_status,
            upstream_failure_reason=args.upstream_reason,
        )
    atomic_json(state_path, state)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        summaries = []
        for identifier in identifiers:
            state.update(current_experiment=identifier, worker_pids=[], updated_at=utcnow())
            atomic_json(state_path, state)
            jobs = [job_spec(config, identifier, seed, project, args.stage) for seed in SEEDS]
            run_group(jobs, state_path, state)
            if args.stage == "formal":
                summaries.append(summarize_experiment(project, identifier))
            state["completed_experiments"].append(identifier)
            state.update(worker_pids=[], updated_at=utcnow())
            atomic_json(state_path, state)
        if args.stage == "formal":
            summarize_chain(project, summaries)
        state.update(status="completed", completed_at=utcnow(), current_experiment=None, worker_pids=[])
        atomic_json(state_path, state)
    except BaseException as error:
        stop_processes(ACTIVE_PROCESSES)
        status = "cancelled" if CANCEL_REQUESTED or isinstance(error, KeyboardInterrupt) else "failed"
        state.update(
            status=status,
            failed_at=utcnow(),
            failure_reason=str(error),
            traceback=traceback.format_exc(),
            worker_pids=[],
        )
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
