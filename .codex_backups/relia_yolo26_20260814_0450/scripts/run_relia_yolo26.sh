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
SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
PYTHON=/home/room305/miniconda3/envs/yolov26/bin/python3
DATA=/home/room305/ZZF/URPC2019/data.yaml
WEIGHTS=/home/room305/ZZF/yolov26/yolo26n.pt
STATS="$PROJECT_ROOT/runs/relia_yolo26_preflight/train_statistics.json"
RUN_ROOT="$PROJECT_ROOT/runs/relia_yolo26_$RUN_ID"
SMOKE_ROOT="$PROJECT_ROOT/runs/relia_yolo26_smoke_$RUN_ID"
STATE="$RUN_ROOT/state.json"

[[ $PROJECT_ROOT == /* && $UPSTREAM_STATE == /* && $RUN_ROOT == /* ]] || { echo "All project/state paths must be absolute." >&2; exit 2; }
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid run ID: $RUN_ID" >&2; exit 2; }
case "$UPSTREAM_STATUS" in completed|failed|cancelled) ;; *) echo "Invalid upstream status: $UPSTREAM_STATUS" >&2; exit 2;; esac
[[ -x $PYTHON && -f $DATA && -f $WEIGHTS && -f $STATS && -f $UPSTREAM_STATE ]] || { echo "Missing interpreter, data, weights, statistics, or upstream state." >&2; exit 1; }
ACTUAL_STATUS=$("$PYTHON" -c "import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')).get('status',''))" "$UPSTREAM_STATE" 2>/dev/null || true)
[[ $ACTUAL_STATUS == "$UPSTREAM_STATUS" ]] || { echo "Upstream state mismatch: ${ACTUAL_STATUS:-invalid}" >&2; exit 1; }
[[ ! -e $STATE || $("$PYTHON" -c "import json,sys; p=json.load(open(sys.argv[1],encoding='utf-8')); print(int(p.get('status')=='waiting' and p.get('run_id')==sys.argv[2]))" "$STATE" "$RUN_ID") == 1 ]] || { echo "Immutable run already exists: $STATE" >&2; exit 1; }

mkdir -p "$RUN_ROOT/train" "$RUN_ROOT/test"
printf '%s\n' "$$" > "$RUN_ROOT/launcher.pid"

mark_smoke_failure() {
    "$PYTHON" - "$STATE" "$SMOKE_ROOT/state.json" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

state_path, smoke_path = map(Path, sys.argv[1:])
state = json.loads(state_path.read_text(encoding="utf-8"))
smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
state.update(
    status="failed",
    phase="smoke",
    failure_reason=smoke.get("failure_reason", "RELiA smoke check failed."),
    updated_at=datetime.now(timezone.utc).isoformat(),
)
temporary = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, state_path)
PY
}

cd /tmp
export CUDA_VISIBLE_DEVICES=0
if [[ ! -e $SMOKE_ROOT/state.json ]]; then
    echo "[$(date -u +%FT%TZ)] starting three-process one-epoch RELiA A7 smoke check"
    if ! "$PYTHON" "$SOURCE_ROOT/tools/urpc/run_relia_ablation.py" \
        --stage smoke --run-id "smoke_$RUN_ID" --project "$SMOKE_ROOT" --data "$DATA" --weights "$WEIGHTS" --relia-stats "$STATS" --ids A7; then
        mark_smoke_failure
        exit 1
    fi
fi
SMOKE_STATUS=$("$PYTHON" -c "import json,sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')).get('status',''))" "$SMOKE_ROOT/state.json")
if [[ $SMOKE_STATUS != completed ]]; then
    mark_smoke_failure
    echo "RELiA smoke check did not complete: $SMOKE_STATUS" >&2
    exit 1
fi
exec "$PYTHON" "$SOURCE_ROOT/tools/urpc/run_relia_ablation.py" \
    --stage formal --run-id "$RUN_ID" --project "$RUN_ROOT" --data "$DATA" --weights "$WEIGHTS" --relia-stats "$STATS" \
    --upstream-state "$UPSTREAM_STATE" --upstream-status "$UPSTREAM_STATUS" --upstream-reason "$UPSTREAM_REASON" \
    --ids A0,A1,A2,A3,A4,A5,A6,A7
