#!/usr/bin/env bash
# Upload the bulk corpus built by scripts/r2-build-corpus.py to R2.
#
#   set -a && . ./.env && set +a          # CLOUDFLARE_API_TOKEN
#   scripts/r2-upload-corpus.sh dist/r2
#
# THE TOKEN NEEDS "Workers R2 Storage: Edit" ON THIS ACCOUNT. The deploy token
# is deliberately narrow (CLAUDE.md), so this permission has to be added
# explicitly -- it is not implied by Workers edit. Without it every call fails
# with "Authentication error [code: 10000]", which does not name the missing
# scope.
#
# Idempotent: re-uploading a key replaces it. The Worker serves these
# immutable-with-a-long-max-age, so a REPLACED file keeps its old name and any
# client holding a cached copy will not see the new one until the cache expires
# -- at a data release, publish under a new manifest and let the filenames
# change, or accept the lag.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=${1:-dist/r2}
BUCKET=${BUCKET:-qmd-bulk}

[ -f "$DIR/manifest.json" ] || { echo "no manifest in $DIR -- run r2-build-corpus.py first" >&2; exit 1; }

# Data files first, manifest LAST: the manifest is what the site advertises, so
# publishing it before the files it lists would offer downloads that 404.
files=$(ls "$DIR" | grep -v '^manifest\.json$')
for f in $files; do
  echo "==> $f ($(du -h "$DIR/$f" | cut -f1))"
  npx wrangler r2 object put "$BUCKET/$f" --file "$DIR/$f" --remote \
    --content-type "$([[ $f == *.gz ]] && echo application/gzip || echo application/json)"
done
echo "==> manifest.json (last: it advertises the files above)"
npx wrangler r2 object put "$BUCKET/manifest.json" --file "$DIR/manifest.json" --remote \
  --content-type application/json

echo
echo "uploaded. verify:  curl -s https://quivermutationdb.org/api/bulk | head -c 400"
