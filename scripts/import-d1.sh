#!/usr/bin/env bash
# Import a per-rank D1 export (dist/d1) in the only order that is correct:
# ranks ascending, parts ascending. Each rank's part 001 deletes that rank's
# rows first, so a rank is idempotent as a whole but NOT part by part.
#
#   scripts/import-d1.sh dist/d1            # local dev database
#   scripts/import-d1.sh dist/d1 --remote   # production
#   scripts/import-d1.sh dist/d1 --remote --ranks 4,5
set -euo pipefail
DIR=${1:?usage: import-d1.sh DIR [--remote] [--ranks a,b]}; shift
MODE=--local; RANKS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --remote) MODE=--remote;;
    --local) MODE=--local;;
    --ranks) RANKS=$2; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac; shift
done
MANIFEST="$DIR/manifest.json"
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 1; }
python3 - "$MANIFEST" "$RANKS" <<'PY' | while read -r f; do
import json, sys
m = json.load(open(sys.argv[1])); want = {int(x) for x in sys.argv[2].split(",") if x}
for n in sorted(int(k) for k in m["ranks"]):
    if want and n not in want: continue
    for part in m["ranks"][str(n)]["parts"]:
        print(part["file"])
PY
  echo ">> $f"
  npx wrangler d1 execute qmd $MODE --file="$DIR/$f"
done
echo "done ($MODE)"
