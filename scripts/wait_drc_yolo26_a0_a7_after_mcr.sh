#!/usr/bin/env bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <absolute_project_root> <immutable_run_id>" >&2
    exit 2
fi

PROJECT_ROOT=$1
RUN_ID=$2
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=/home/room305/miniconda3/envs/yolov26/bin/python
UPSTREAM_STATE=/home/room305/ZZF/yolov13-6000/runs/mcr_urpc2019_20260810_105100/state.json
RUN_ROOT="$PROJECT_ROOT/runs/drc_yolo26_a0_a7_$RUN_ID"
STATE="$RUN_ROOT/state.json"

[[ $PROJECT_ROOT == /* ]] || { echo "Project root must be absolute: $PROJECT_ROOT" >&2; exit 2; }
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid run ID: $RUN_ID" >&2; exit 2; }
[[ -x $PYTHON ]] || { echo "Missing Python interpreter: $PYTHON" >&2; exit 1; }
[[ ! -e $STATE ]] || { echo "Run ID already exists: $STATE" >&2; exit 1; }
mkdir -p "$RUN_ROOT/train" "$RUN_ROOT/test"
printf '%s\n' "$$" > "$RUN_ROOT/launcher.pid"

write_wait_state() {
    "$PYTHON" - "$STATE" "$RUN_ID" "$UPSTREAM_STATE" "$1" "$2" "$PROJECT_ROOT" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "scheme": "drc_yolo26_a0_a7",
    "run_id": sys.argv[2],
    "stage": "formal",
    "status": "waiting",
    "phase": "waiting_mcr",
    "upstream_state": sys.argv[3],
    "upstream_status": sys.argv[4],
    "upstream_failure_reason": sys.argv[5],
    "run_root": str(Path(sys.argv[6]) / "runs" / f"drc_yolo26_a0_a7_{sys.argv[2]}"),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

echo "[$(date -u +%FT%TZ)] waiting for MCR state: $UPSTREAM_STATE"
while true; do
    state_line=$("$PYTHON" - "$UPSTREAM_STATE" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    status = str(payload.get("status", ""))
    reason = str(payload.get("failure_reason", payload.get("reason", ""))).replace("\n", " ")
    print(f"{status}|{reason}")
except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
    print("invalid|")
PY
)
    upstream_status=${state_line%%|*}
    upstream_reason=${state_line#*|}
    case "$upstream_status" in
        completed|failed|cancelled)
            echo "[$(date -u +%FT%TZ)] MCR terminal state=$upstream_status reason=$upstream_reason"
            write_wait_state "$upstream_status" "$upstream_reason"
            exec bash "$SOURCE_ROOT/scripts/run_drc_yolo26_a0_a7.sh" "$PROJECT_ROOT" "$RUN_ID" "$UPSTREAM_STATE" "$upstream_status" "$upstream_reason"
            ;;
        *)
            write_wait_state "$upstream_status" "$upstream_reason"
            sleep 30
            ;;
    esac
done
