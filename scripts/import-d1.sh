#!/usr/bin/env bash
# Import a per-rank D1 export (dist/…) in the only order that is correct:
# ranks ascending, parts ascending, each part into the database the manifest
# names (split ranks span several databases; rank_stats goes to the main one).
# Each shard's part 001 deletes that rank's rows there first, so a rank is
# idempotent as a whole but NOT part by part.
#
#   scripts/import-d1.sh dist/d1                    # local dev databases
#   scripts/import-d1.sh dist/d1 --remote           # production
#   scripts/import-d1.sh dist/d1 --remote --ranks 4,5
#   scripts/import-d1.sh dist/d1 --remote --ranks 6 --shard n6.0   # one shard (stage big loads)
set -euo pipefail
DIR=${1:?usage: import-d1.sh DIR [--remote] [--ranks a,b] [--shard key]}; shift
MODE=--local; RANKS=""; SHARD=""
while [ $# -gt 0 ]; do
  case "$1" in
    --remote) MODE=--remote;;
    --local) MODE=--local;;
    --ranks) RANKS=$2; shift;;
    --shard) SHARD=$2; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac; shift
done
MANIFEST="$DIR/manifest.json"
[ -f "$MANIFEST" ] || { echo "no $MANIFEST" >&2; exit 1; }
python3 - "$MANIFEST" "$RANKS" "$SHARD" <<'PY' | while read -r db f; do
import json, sys
m = json.load(open(sys.argv[1])); want = {int(x) for x in sys.argv[2].split(",") if x}; shard = sys.argv[3]
for n in sorted(int(k) for k in m["ranks"]):
    if want and n not in want: continue
    for part in m["ranks"][str(n)]["parts"]:
        if shard and part.get("shard") != shard and not part["file"].endswith(".stats.001.sql"): continue
        print(part.get("database", "qmd"), part["file"])
PY
  echo ">> $db <- $f"
  npx wrangler d1 execute "$db" $MODE --file="$DIR/$f"
done
echo "done ($MODE)"
