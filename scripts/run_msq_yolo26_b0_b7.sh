#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <absolute_project_root> <immutable_run_id> <absolute_upstream_state> <upstream_status> <upstream_reason>" >&2
    exit 2
fi

PROJECT_ROOT=$1
RUN_ID=$2
UPSTREAM_STATE=$3
UPSTREAM_STATUS=$4
UPSTREAM_REASON=$5
SOURCE_ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
PYTHON=/home/room305/miniconda3/envs/yolov26/bin/python
DATA=/home/room305/ZZF/URPC2019/data.yaml
WEIGHTS="$PROJECT_ROOT/yolo26n.pt"
RUN_ROOT="$PROJECT_ROOT/runs/msq_yolo26_b0_b7_$RUN_ID"
LAUNCHER_LOG="$RUN_ROOT/train/launcher.log"
STATE="$RUN_ROOT/state.json"

[[ $PROJECT_ROOT == /* && $UPSTREAM_STATE == /* ]] || { echo "Project and upstream state paths must be absolute" >&2; exit 2; }
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid run ID: $RUN_ID" >&2; exit 2; }
case "$UPSTREAM_STATUS" in completed|failed|cancelled) ;; *) echo "Invalid upstream status: $UPSTREAM_STATUS" >&2; exit 2;; esac
[[ -x $PYTHON ]] || { echo "Missing Python interpreter: $PYTHON" >&2; exit 1; }
[[ -f $DATA ]] || { echo "Missing dataset YAML: $DATA" >&2; exit 1; }
[[ -f $WEIGHTS ]] || { echo "Missing YOLO26-N weights: $WEIGHTS" >&2; exit 1; }
[[ -f $UPSTREAM_STATE ]] || { echo "Missing upstream state: $UPSTREAM_STATE" >&2; exit 1; }

ACTUAL_STATUS=$("$PYTHON" -c "import json, sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')).get('status', ''))" "$UPSTREAM_STATE" 2>/dev/null || true)
[[ $ACTUAL_STATUS == "$UPSTREAM_STATUS" ]] || { echo "Upstream state mismatch: $ACTUAL_STATUS" >&2; exit 1; }
if [[ -e $STATE ]]; then
    "$PYTHON" -c "import json, sys; p=json.load(open(sys.argv[1], encoding='utf-8')); raise SystemExit(0 if p.get('status') == 'waiting' and p.get('run_id') == sys.argv[2] else 1)" "$STATE" "$RUN_ID" \
        || { echo "Run ID already exists: $STATE" >&2; exit 1; }
fi

mkdir -p "$RUN_ROOT/train" "$RUN_ROOT/test"
[[ -e $LAUNCHER_LOG ]] || : > "$LAUNCHER_LOG"
printf '%s\n' "$$" > "$RUN_ROOT/launcher.pid"

cd /tmp
export CUDA_VISIBLE_DEVICES=0
exec "$PYTHON" "$SOURCE_ROOT/tools/urpc/run_ablation.py" \
    --scheme msq \
    --stage formal \
    --run-id "$RUN_ID" \
    --data "$DATA" \
    --pretrained "$WEIGHTS" \
    --project "$RUN_ROOT" \
    --upstream-state "$UPSTREAM_STATE" \
    --upstream-status "$UPSTREAM_STATUS" \
    --upstream-reason "$UPSTREAM_REASON" \
    --ids B0,B1,B2,B3,B4,B5,B6,B7
