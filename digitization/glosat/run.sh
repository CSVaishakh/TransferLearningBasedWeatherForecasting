#!/usr/bin/env bash
# Run GloSAT table detection + table structure recognition on a folder of images.
#
#   ./run.sh <input_image_dir> <output_dir> [coarse|fine]
#
# coarse = table + coarse cell regions (rows/columns blocks)
# fine   = table + individual cells, assisted by the coarse model (default)
#
# Outputs (in <output_dir>): ICDAR-19 style XML per image, and annotated images
# with the predicted boxes drawn on them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GLOSAT_REPO="${GLOSAT_REPO:-$HERE/../../../glosat_table_dataset}"
MODELS="${MODELS:-$GLOSAT_REPO/models}"

IN="$(realpath "$1")"
OUT="$(realpath -m "$2")"
MODE="${3:-fine}"
mkdir -p "$OUT"

TABLE=/models/model_fulltables_only_GloSAT.pth
COARSE=/models/model_coarsecell_GloSAT.pth
FINE=/models/model_finecell_GloSAT.pth

for f in "$MODELS"/model_fulltables_only_GloSAT.pth "$MODELS"/model_coarsecell_GloSAT.pth "$MODELS"/model_finecell_GloSAT.pth; do
  [ -f "$f" ] || { echo "missing checkpoint: $f (unzip models.zip into $MODELS first)"; exit 1; }
done

if [ "$MODE" = coarse ]; then
  CELLS=(--cell_checkpoint "$COARSE")
else
  CELLS=(--cell_checkpoint "$FINE" --coarse_cell_checkpoint "$COARSE")
fi

docker run --rm \
  -v "$MODELS":/models:ro \
  -v "$IN":/data/in:ro \
  -v "$OUT":/data/out \
  -e GLOSAT_DEVICE=cpu \
  glosat-cpu \
  python dla/src/inference.py "$TABLE" "${CELLS[@]}" \
    --load_from /data/in --out /data/out/ --visual True
