#!/usr/bin/env bash
# Apply the drizzle migrations to EVERY shard database (data/shards.json).
#   scripts/migrate-all.sh            # local
#   scripts/migrate-all.sh --remote   # production
set -euo pipefail
MODE=${1:---local}
for db in $(python3 -c 'import json; c=json.load(open("data/shards.json")); print(c["main"]["database"]); [print(d["database"]) for s in c["split"].values() for d in s["databases"]]'); do
  echo ">> migrating $db ($MODE)"
  npx wrangler d1 migrations apply "$db" $MODE
done
