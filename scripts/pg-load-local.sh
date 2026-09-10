#!/usr/bin/env bash
# Build the dev dataset and load it into a Postgres database.
#
# This is the D1 `migrate-all.sh + import-d1.sh` pair replaced. It is the same
# path production took (docs/PLANETSCALE.md, "Phase 2 as it actually ran"), just
# on a dataset small enough for CI: generate -> D1 SQL parts -> one SQLite ->
# COPY text -> psql. Going through SQLite is not a detour: pg-export.py reads a
# QMD SQLite, and reusing it keeps CI honest about the real load path rather
# than inventing a second one that could drift.
#
# Connection comes from libpq's own PG* variables (or PGURL). Usage:
#   set -a && . ./.pgenv && set +a && scripts/pg-load-local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# NEVER dist/d1. That directory holds the real multi-rank production export
# (~7 GB, ranks 1-8, and the acyclicity-n{j}.json checkpoints every higher rank
# depends on BY SHA256). Generating the dev cell into it silently overwrites
# ranks 1-4 and invalidates the manifest's depends_on chain for everything
# above them -- which is exactly what happened once, on 2026-09-09, before this
# comment existed. The dev dataset is a build artifact; keep it somewhere it can
# be deleted without thinking.
D1="${D1:-dist/dev-d1}"
OUT="${OUT:-dist/dev-pg}"
SQLITE="${SQLITE:-dist/qmd-dev.sqlite}"
PSQL=(psql -X -v ON_ERROR_STOP=1 -q)

# REFUSE TO RUN AGAINST ANYTHING BUT A LOCAL DATABASE.
# This script drops and recreates the public schema. .pgenv points at the
# PRODUCTION PlanetScale database, and sourcing it before running this -- the
# obvious muscle memory, since every other psql command here needs it -- would
# destroy the live census. Localhost only unless the caller says otherwise in
# so many words.
HOST_CHECK="${PGHOST:-localhost}"
case "$HOST_CHECK" in
  localhost|127.0.0.1|::1|/*) ;;
  *)
    if [ "${I_KNOW_THIS_IS_NOT_LOCAL:-}" != "yes" ]; then
      echo "REFUSING: PGHOST=$HOST_CHECK is not local, and this script runs" >&2
      echo "  DROP SCHEMA public CASCADE. If you are certain, re-run with" >&2
      echo "  I_KNOW_THIS_IS_NOT_LOCAL=yes" >&2
      exit 1
    fi
    echo "!! running against NON-LOCAL host $HOST_CHECK on an explicit override" >&2
    ;;
esac

echo "==> generating the dev dataset into $D1"
rm -rf "$D1"
python scripts/populate.py --export-d1 "$D1"

echo "==> building one SQLite from the D1 parts"
rm -f "$SQLITE"
for m in drizzle/*.sql; do sqlite3 "$SQLITE" < "$m"; done
# Ranks ascending, parts in order. Every rank goes into this one SQLite: the
# per-shard split exists only because D1 had a size ceiling, and part 001 of a
# rank deletes that rank first, so ordering is what keeps it idempotent.
python - "$SQLITE" "$D1" <<'PY'
import json, sqlite3, subprocess, sys
sqlite_path = sys.argv[1]
d1 = sys.argv[2]
manifest = json.load(open(f"{d1}/manifest.json"))
for rank in sorted(manifest["ranks"], key=int):
    for part in manifest["ranks"][rank]["parts"]:
        subprocess.run(["sqlite3", sqlite_path], check=True,
                       stdin=open(f"{d1}/{part['file']}"))
con = sqlite3.connect(sqlite_path)
print("    loaded:", dict(con.execute("SELECT n, count(*) FROM quivers GROUP BY n ORDER BY n")))
PY

echo "==> rendering Postgres COPY text"
rm -rf "$OUT"
python scripts/pg-export.py "$OUT" "$SQLITE"

echo "==> loading into Postgres"
"${PSQL[@]}" -c 'DROP SCHEMA public CASCADE; CREATE SCHEMA public;'
"${PSQL[@]}" -f drizzle-pg/0001_init.sql
"${PSQL[@]}" -f "$OUT/load.sql"
"${PSQL[@]}" -f drizzle-pg/0002_indexes.sql
"${PSQL[@]}" -f drizzle-pg/0003_rank8_dangling_class_refs.sql
python scripts/nicknames.py --sql dist/nicknames.sql
"${PSQL[@]}" -f dist/nicknames.sql
"${PSQL[@]}" -c 'ANALYZE'

# Build a small bulk corpus from what we just loaded and put it in the LOCAL R2
# simulation, so `npm run test:api` exercises /api/bulk against real objects
# rather than skipping it. Without this the bulk path would be untested until
# production, which is where its bugs would then be found.
echo "==> building the bulk corpus and loading it into local R2"
CORPUS="${CORPUS:-dist/dev-r2}"
rm -rf "$CORPUS"
python scripts/r2-build-corpus.py "$CORPUS" \
  --database "${PGDATABASE:-qmd}" --host "${PGHOST:-127.0.0.1}"
for f in "$CORPUS"/*; do
  npx wrangler r2 object put "qmd-bulk/$(basename "$f")" --file "$f" --local >/dev/null
done
echo "    $(ls "$CORPUS" | wc -l | tr -d ' ') objects in local R2"

echo "==> loaded"
psql -X -tAc "SELECT '    rank ' || n || ': ' || count(*) FROM quivers GROUP BY n ORDER BY n"
