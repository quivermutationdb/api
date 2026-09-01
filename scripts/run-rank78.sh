#!/usr/bin/env bash
# Generate ranks 7 and 8 (the last two cells of the census), detached.
#
#   (7, 1)  2,120,098 connected quivers — enumerated exactly (orderly)
#   (8, 1)  572,849,763 connected quivers — far too many, so a 1 M sample
#
# Sequential because rank 8 consumes rank 7's acyclicity-n7.json checkpoint,
# and two invocations because the two cells use different generators.
#
# Both runs also get the constructed seeds (qmd/d1_export._curated_seeds):
# every Dynkin and affine-E type of the rank plus one quiver per triangulated
# surface. A cell at |b_ij| <= 1 cannot contain a mutation-finite class whose
# every member has a double arrow, and rank 8 is sampled besides — so those
# classes are put in deliberately rather than left to luck. Exploration runs at
# EXPLORE_BOUND = 2, so each seed drags its whole class in, double arrows and
# all; expect quiver_count to exceed the cell size for that reason.
#
# As with rank 6: run-detached.sh applies `caffeinate -s` (survives a shut lid
# on AC power), and the pipeline is pure stdlib so it runs on the native arm64
# interpreter rather than the Rosetta-translated venv (~1.6x faster, and the
# golden ids are identical on both).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${QMD_PYTHON:-/usr/bin/python3}
LOG=${QMD_LOG:-dist/d1/census-r78.log}
COMMON="--export-d1 dist/d1 --max-vertices 8 --bound 1 --node-cap 100 --workers 8 --la-timeout 1.0"
mkdir -p "$(dirname "$LOG")"

"$PY" -c "import sys; sys.path.insert(0,'.'); import qmd.core, qmd.surfaces" \
  || { echo "cannot import qmd with $PY" >&2; exit 1; }

echo "generating ranks 7 then 8 with $PY -> $LOG"
exec scripts/run-detached.sh "$LOG" /bin/bash -c "
cd '$PWD'
set -e
echo '=== rank 7 (orderly, exact cell) ==='
$PY scripts/populate.py $COMMON --ranks 7
echo '=== rank 8 (1M sample) ==='
$PY scripts/populate.py $COMMON --ranks 8 --generator sample --sample 1000000
echo '=== done ==='
"
