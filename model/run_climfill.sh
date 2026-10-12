#!/usr/bin/env bash
# Stage-3 runs with the SJCET climatology fill (train.py --clim-fill), same fractions/seeds as
# run_stage3.sh, for the pretrained 1->2->3 model and the from-scratch model. Then export.
#   ./run_climfill.sh            (resumable: finished runs are skipped)
set -euo pipefail
cd "$(dirname "$0")"
SEEDS="${SEEDS:-0 1 2}"
FRACTIONS="${FRACTIONS:-0.1 0.25 0.5 1.0}"
run() {
  local id=$1; shift
  if [ -f "outputs/runs/$id/result.json" ]; then echo "skip $id"; return; fi
  echo "=== $id ($(date +%H:%M))"
  uv run python -W ignore train.py --run-id "$id" --clim-fill "$@"
}
[ -f outputs/runs/s12/best.pt ] || { echo "missing outputs/runs/s12/best.pt: run stages 1-2 first"; exit 1; }
for f in $FRACTIONS; do
  for s in $SEEDS; do
    if [ "$f" = "1.0" ] && [ "$s" != "0" ]; then continue; fi
    run "s123c_f${f}_s${s}"    --stage 3 --fraction "$f" --seed "$s" --init outputs/runs/s12/best.pt
    run "scratchc_f${f}_s${s}" --stage 3 --fraction "$f" --seed "$s"
  done
done
uv run python -W ignore export.py
echo "ALL DONE $(date +%H:%M)"
