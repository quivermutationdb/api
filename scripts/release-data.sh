#!/usr/bin/env bash
# Production data release to PlanetScale Postgres, in the only safe order:
#   1. verify the export   2. schema   3. rank data   4. indexes
#   5. patches + nicknames 6. verify   7. deploy the Worker
#
# The Worker is deployed LAST: new code may expect data the release adds, and a
# Worker deployed first would serve against a half-loaded database.
#
#   set -a && . ./.pgenv && set +a          # Postgres connection (NOT .env --
#   set -a && . ./.env   && set +a          # see .pgenv for why they are split)
#   scripts/release-data.sh dist/d1
#
# NOT for incremental updates: 0001_init.sql creates the schema from nothing, so
# this is a full rebuild. Loading a single new rank into a live database is a
# different job -- follow docs/PLANETSCALE.md, "Phase 2 as it actually ran", and
# chunk anything the size of rank 6.
set -euo pipefail
DIR=${1:?usage: release-data.sh DIR}
OUT="${OUT:-dist/pg-release}"
SQLITE="${SQLITE:-dist/qmd-release.sqlite}"
PSQL=(psql -X -v ON_ERROR_STOP=1)

echo "== 0/7 verify export (connected-only guarantee)"
python3 scripts/verify-export.py "$DIR"

echo "== 1/7 build one SQLite from the D1 parts, then Postgres COPY text"
rm -f "$SQLITE"; rm -rf "$OUT"
for m in drizzle/*.sql; do sqlite3 "$SQLITE" < "$m"; done
python3 - "$SQLITE" "$DIR" <<'PY'
import json, subprocess, sys
sqlite_path, d = sys.argv[1], sys.argv[2]
manifest = json.load(open(f"{d}/manifest.json"))
for rank in sorted(manifest["ranks"], key=int):
    for part in manifest["ranks"][rank]["parts"]:
        subprocess.run(["sqlite3", sqlite_path], check=True, stdin=open(f"{d}/{part['file']}"))
PY
python3 scripts/pg-export.py "$OUT" "$SQLITE"

echo "== 2/7 schema (no indexes, not even primary keys -- bare heap for the load)"
"${PSQL[@]}" -f drizzle-pg/0001_init.sql

echo "== 3/7 rank data"
"${PSQL[@]}" -f "$OUT/load.sql"

echo "== 4/7 indexes, primary keys, foreign key (CONCURRENTLY; the slow step)"
"${PSQL[@]}" -f drizzle-pg/0002_indexes.sql

echo "== 5/7 data patches + curated nicknames"
"${PSQL[@]}" -f drizzle-pg/0003_rank8_dangling_class_refs.sql
python3 scripts/nicknames.py --sql "$DIR/nicknames.sql"
"${PSQL[@]}" -f "$DIR/nicknames.sql"
"${PSQL[@]}" -c 'ANALYZE'

echo "== 6/7 verify (a nonzero orphan count or a false here means STOP)"
"${PSQL[@]}" -f scripts/pg-verify.sql
"${PSQL[@]}" -f scripts/pg-verify-census.sql

echo "== 7/7 deploy"
npm run deploy
echo "== live"; curl -s https://quivermutationdb.org/api/stats | head -c 400; echo
