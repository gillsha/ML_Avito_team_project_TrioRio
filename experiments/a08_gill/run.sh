#!/usr/bin/env bash
# Run from any directory, with the project's pinned Python on PATH.
set -euo pipefail
export PYTHONHASHSEED=0
root="$(cd "$(dirname "$0")" && pwd)"
if [[ $# != 4 ]]; then
  echo "Usage: $0 DATA_DIR MODEL_DIR RESULTS_DIR HARDWARE_DESCRIPTION" >&2
  exit 2
fi
data=$1
model=$2
results=$3
hardware=$4
mkdir -p "$results"
starter="$root/course/far-trans/starter.py"
/usr/bin/time -p python "$starter" run --data-dir "$data" --output-dir "$results/baseline" > "$results/baseline.log" 2>&1
python "$root/rank_a08.py" prepare --data-dir "$data" --baseline-dir "$results/baseline" --output-dir "$results"
python "$root/analyze_a08.py" audit --data-dir "$data" --results "$results"
/usr/bin/time -p python "$root/rank_a08.py" infer --output-dir "$results" --model-dir "$model" \
  --hardware "$hardware" --pilot-users 3 --users-per-batch 4 2>&1 | tee "$results/pilot.log"
python - "$results" <<'PY'
import json
import sys
from pathlib import Path

for variant in ('category', 'original'):
    path = Path(sys.argv[1]) / 'pilot' / f'{variant}_responses.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != 3 or not all(record['raw_response'] for record in records):
        sys.exit(f'Pilot incomplete: {path}')
    valid = sum(not record['fallback'] for record in records)
    print(f'Pilot {variant}: {valid}/3 valid replies; fallback reasons: '
          f'{[r["error"] for r in records if r["fallback"]]}')
PY
/usr/bin/time -p python "$root/rank_a08.py" infer --output-dir "$results" --model-dir "$model" \
  --hardware "$hardware" --users-per-batch 4 2>&1 | tee "$results/inference.log"
for variant in category original; do
  /usr/bin/time -p python "$starter" run --data-dir "$data" --output-dir "$results/$variant" \
    --ranker-candidate-source popularity --on-invalid error \
    --responses "$results/inference/${variant}_responses.jsonl" > "$results/$variant.log" 2>&1
done
python "$root/analyze_a08.py" analyze --data-dir "$data" --results "$results"
