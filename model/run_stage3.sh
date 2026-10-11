#!/usr/bin/env bash
# Stage-3 fine-tuning on the local station only, reusing existing stage-1/2
# checkpoints (outputs/runs/s1, s12): baselines, the data-efficiency ablation
# (3 strategies x 4 fractions x seeds), then export.
#   ./run_stage3.sh            (resumable: finished runs are skipped)
#   SEEDS="0" ./run_stage3.sh  (quicker)
set -euo pipefail
cd "$(dirname "$0")"
SEEDS="${SEEDS:-0 1 2}"
FRACTIONS="${FRACTIONS:-0.1 0.25 0.5 1.0}"
run() {  # run <run-id> <args...>; skip if result exists
  local id=$1; shift
  if [ -f "outputs/runs/$id/result.json" ]; then echo "skip $id"; return; fi
  echo "=== $id ($(date +%H:%M))"
  uv run python -W ignore train.py --run-id "$id" "$@"
}

for f in outputs/runs/s1/best.pt outputs/runs/s12/best.pt; do
  [ -f "$f" ] || { echo "missing $f: run ./run_all.sh first"; exit 1; }
done
[ -f outputs/baselines.json ] || uv run python -W ignore baselines.py --seeds $SEEDS

for f in $FRACTIONS; do
  for s in $SEEDS; do
    if [ "$f" = "1.0" ] && [ "$s" != "0" ]; then continue; fi  # full data: one subset only
    run "scratch_f${f}_s${s}" --stage 3 --fraction "$f" --seed "$s"
    run "s13_f${f}_s${s}"     --stage 3 --fraction "$f" --seed "$s" --init outputs/runs/s1/best.pt
    run "s123_f${f}_s${s}"    --stage 3 --fraction "$f" --seed "$s" --init outputs/runs/s12/best.pt
  done
done
uv run python -W ignore export.py
uv run python -W ignore predict.py --export
echo "ALL DONE $(date +%H:%M)"
