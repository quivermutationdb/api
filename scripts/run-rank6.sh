#!/usr/bin/env bash
# Resume (or start) the rank-6 streaming census.
#
# Everything already in dist/d1/work-n6.sqlite is kept: `generate` and
# `invariants` are marked complete, and `label` resumes from its committed
# watermark. Safe to run repeatedly.
#
# Two things this wrapper exists to get right, both of which cost us days:
#
#  * Keep the machine on AC power. run-detached.sh applies `caffeinate -s`,
#    which prevents system sleep on AC — `caffeinate -i` blocks only *idle*
#    sleep and a shut lid suspended the job anyway, costing three days.
#  * The pipeline is pure stdlib, so it runs on the system's native arm64
#    Python. The x86_64 venv Python runs under Rosetta at ~0.6x, and the
#    golden n<=4 ids are byte-identical on both (verified), so there is no
#    re-keying risk in using it here.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${QMD_PYTHON:-/usr/bin/python3}
LOG=${QMD_LOG:-dist/d1/census-r6.log}
mkdir -p "$(dirname "$LOG")"

"$PY" -c "import sys; sys.path.insert(0,'.'); import qmd.core" \
  || { echo "cannot import qmd with $PY" >&2; exit 1; }

echo "resuming rank 6 with $PY -> $LOG"
exec scripts/run-detached.sh "$LOG" "$PY" -c "
import sys; sys.path.insert(0, '.')
from qmd.bigcell import export_big_cell
export_big_cell('dist/d1', n=6, h=2, label_cap=20, node_cap=100,
                sample=250_000, workers=8, la_timeout=0.0)
"
