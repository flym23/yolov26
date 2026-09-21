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
PYTHON=/home/room305/miniconda3/envs/yolov26/bin/python3
UPSTREAM_STATE=/home/room305/ZZF/yolov13-6000/runs/dor_urpc2019_20260812_051119/state.json
RUN_ROOT="$PROJECT_ROOT/runs/relia_yolo26_$RUN_ID"
STATE="$RUN_ROOT/state.json"

[[ $PROJECT_ROOT == /* ]] || { echo "Project root must be absolute." >&2; exit 2; }
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid run ID: $RUN_ID" >&2; exit 2; }
[[ -x $PYTHON && -f $UPSTREAM_STATE && ! -e $STATE ]] || { echo "Missing interpreter/upstream state or run ID already exists." >&2; exit 1; }
mkdir -p "$RUN_ROOT/train" "$RUN_ROOT/test"
printf '%s\n' "$$" > "$RUN_ROOT/launcher.pid"

write_state() {
    "$PYTHON" - "$STATE" "$RUN_ID" "$PROJECT_ROOT" "$UPSTREAM_STATE" "$1" "$2" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {"schema_version": 1, "scheme": "relia_yolo26", "run_id": sys.argv[2], "status": "waiting", "phase": "waiting_dor", "run_root": str(Path(sys.argv[3]) / "runs" / f"relia_yolo26_{sys.argv[2]}"), "upstream_state": sys.argv[4], "upstream_status": sys.argv[5], "upstream_failure_reason": sys.argv[6], "updated_at": datetime.now(timezone.utc).isoformat()}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

echo "[$(date -u +%FT%TZ)] waiting for DOR state: $UPSTREAM_STATE"
while true; do
    state_line=$("$PYTHON" - "$UPSTREAM_STATE" <<'PY'
import json, sys
from pathlib import Path
try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(f"{payload.get('status','')}|{str(payload.get('failure_reason', payload.get('reason',''))).replace(chr(10),' ')}")
except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
    print("invalid|")
PY
)
    upstream_status=${state_line%%|*}
    upstream_reason=${state_line#*|}
    case "$upstream_status" in
        completed|failed|cancelled)
            echo "[$(date -u +%FT%TZ)] DOR terminal state=$upstream_status reason=$upstream_reason"
            write_state "$upstream_status" "$upstream_reason"
            exec bash "$SOURCE_ROOT/scripts/run_relia_yolo26.sh" "$PROJECT_ROOT" "$RUN_ID" "$UPSTREAM_STATE" "$upstream_status" "$upstream_reason"
            ;;
        *)
            write_state "$upstream_status" "$upstream_reason"
            sleep 30
            ;;
    esac
done
