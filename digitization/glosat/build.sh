#!/usr/bin/env bash
# Build the CPU-only GloSAT image.
#   GLOSAT_REPO=/path/to/glosat_table_dataset ./build.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GLOSAT_REPO="${GLOSAT_REPO:-$HERE/../../../glosat_table_dataset}"

docker build -t glosat-cpu \
  -f "$HERE/Dockerfile" \
  --build-context patches="$HERE" \
  "$GLOSAT_REPO"
