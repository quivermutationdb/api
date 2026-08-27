#!/usr/bin/env bash
# Drop the quiver indexes a SINGLE-RANK shard does not need, before loading it.
# Inside a rank-6 shard `n` is constant, so (n)-prefixed indexes add nothing
# but row-writes and storage. What stays: the primary key, (n, mutation_finite)
# — the finiteness filter ML users rely on — and (mutation_class_id,
# labeling_count) for class-member listings. Sorting a shard by max_edge or
# filtering by representation type becomes a scan of that shard (~10 M rows,
# a few seconds); acceptable for occasional browsing.
#
#   scripts/trim-shard-indexes.sh            # local
#   scripts/trim-shard-indexes.sh --remote   # production (run once per shard, before its import)
set -euo pipefail
MODE=${1:---local}
for db in $(python3 -c 'import json; c=json.load(open("data/shards.json")); [print(d["database"]) for s in c["split"].values() for d in s["databases"]]'); do
  echo ">> trimming indexes on $db ($MODE)"
  npx wrangler d1 execute "$db" $MODE --command "DROP INDEX IF EXISTS idx_q_n; DROP INDEX IF EXISTS idx_q_n_max_edge; DROP INDEX IF EXISTS idx_q_representation_type;"
done
