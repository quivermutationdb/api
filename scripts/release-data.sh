#!/usr/bin/env bash
# Production data release, in the only safe order (docs/PHASE2.md §8):
#   1. schema migrations      2. rank data (parts in order)
#   3. curated nicknames      4. deploy the Worker
# Run from the repo root with CLOUDFLARE_API_TOKEN in .env:
#   scripts/release-data.sh dist/d1
# The Worker must be deployed AFTER the data: new code expects schema v2.
set -euo pipefail
DIR=${1:?usage: release-data.sh DIR}
echo "== 0/4 verify export (connected-only guarantee)"; python3 scripts/verify-export.py "$DIR"
echo "== 1/4 migrations"; npm run db:migrate:remote
echo "== 2/4 rank data";  scripts/import-d1.sh "$DIR" --remote
echo "== 3/4 nicknames";  python3 scripts/nicknames.py --sql "$DIR/nicknames.sql"
npx wrangler d1 execute qmd --remote --file="$DIR/nicknames.sql"
echo "== 4/4 deploy";     npm run deploy
echo "== verify";         curl -s https://quivermutationdb.org/api/stats | head -c 400; echo
