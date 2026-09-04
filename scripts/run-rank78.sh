#!/usr/bin/env bash
# Generate ranks 7 and 8 through the STREAMING pipeline (qmd/bigcell.py),
# the same way rank 6 was done. Resumable, detached.
#
# Why not scripts/populate.py: export_ranks holds the whole job in memory —
# the seed list, the covered-id set, and every kept orbit WITH its member
# matrices. Rank 7 has ~390k orbits of up to 100 members; the first attempt
# exhausted 16 GB, drove 7 GB of swap, and slowed from 5% in 1h51m to 10% in
# 13h31m. bigcell keeps one row per quiver in scratch SQLite instead and never
# holds the census whole.
#
#   (7, 1)  2,120,098 connected quivers — cell enumerated exactly
#   (8, 1)  572,849,763 connected — far too many, so cell_sample draws 250k
#
# `sample` (the CLASS-row sample) is 45k, not rank 6's 250k, and that is a
# memory budget rather than a preference. run_generation holds every explored
# quiver in memory at 768 bytes each, and a cell bound below EXPLORE_BOUND
# sends orbits outside the cell:
#
#     rank 6, 250k seeds  ->  2,395,384 quivers  ->  1.8 GB   (fine)
#     rank 7, 250k seeds  -> 14,678,008 quivers  -> 11.3 GB   (thrashed 16 GB)
#     rank 7,  45k seeds  -> ~2,642,000 quivers  -> ~2.0 GB   (rank 6's profile)
#
# The finite classes do NOT depend on this sample — they are seeded by
# construction in finite_class_seeds — so a smaller sample costs class-row
# coverage of the infinite classes and nothing else.
#
# Both ranks get, exactly as rank 6 did:
#   * a finiteness verdict for EVERY quiver in the scratch table (label, then
#     stage_resolve for whatever the small cap left unknown)
#   * class rows for a 250k sample
#   * and the guarantee: every Dynkin, affine and surface type of the rank is
#     seeded by construction in finite_class_seeds and stored as a COMPLETE
#     class, with a quiver row for every member — including members outside
#     the cell, which are inserted into the scratch table on the way through.
#
# As with rank 6: run-detached.sh applies `caffeinate -s` (survives a shut lid
# on AC power) and the pipeline is pure stdlib, so it runs on the native arm64
# interpreter rather than the Rosetta-translated venv.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${QMD_PYTHON:-/usr/bin/python3}
LOG=${QMD_LOG:-dist/d1/census-r78.log}
mkdir -p "$(dirname "$LOG")"

"$PY" -c "import sys; sys.path.insert(0,'.'); import qmd.bigcell, qmd.surfaces" \
  || { echo "cannot import qmd with $PY" >&2; exit 1; }

echo "generating ranks 7 then 8 via bigcell with $PY -> $LOG"
exec scripts/run-detached.sh "$LOG" "$PY" -c "
import sys; sys.path.insert(0, '.')
from qmd.bigcell import export_big_cell
print('=== rank 7 (cell enumerated) ===', flush=True)
export_big_cell('dist/d1', n=7, h=1, label_cap=20, node_cap=100,
                sample=45_000, workers=8, la_timeout=0.0)
print('=== rank 8 (cell sampled) ===', flush=True)
export_big_cell('dist/d1', n=8, h=1, label_cap=20, node_cap=100,
                cell_sample=250_000, sample=45_000, workers=8, la_timeout=0.0)
print('=== done ===', flush=True)
"
