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
# Order matters twice over: data files before the manifest, because the
# manifest is what the site advertises and publishing it first would offer
# downloads that 404; and pruning after the manifest, so at no moment does a
# live manifest reference a file that is not in the bucket.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=${1:-dist/r2}
BUCKET=${BUCKET:-qmd-bulk}

[ -f "$DIR/manifest.json" ] || { echo "no manifest in $DIR -- run r2-build-corpus.py first" >&2; exit 1; }

# The manifest ALREADY IN THE BUCKET is the inventory of what is currently
# published; fetch it before overwriting. A rank can change shape between
# releases -- rank 6 outgrew the 315 MB single-object ceiling and became
# numbered parts -- and a leftover qmd-n6.ndjson.gz would otherwise sit there
# being served and being wrong. There is no `wrangler r2 object list`, so this
# is the only inventory available.
OLD=$(mktemp)
if npx wrangler r2 object get "$BUCKET/manifest.json" --file "$OLD" --remote >/dev/null 2>&1; then
  echo "==> comparing against the manifest currently published"
else
  echo "==> no manifest in the bucket yet (first publish)"
  : > "$OLD"
fi

for f in $(ls "$DIR" | grep -v '^manifest\.json$'); do
  echo "==> $f ($(du -h "$DIR/$f" | cut -f1))"
  ct=application/json
  case "$f" in *.gz) ct=application/gzip ;; esac
  npx wrangler r2 object put "$BUCKET/$f" --file "$DIR/$f" --remote --content-type "$ct"
done

echo "==> manifest.json (last: it advertises the files above)"
npx wrangler r2 object put "$BUCKET/manifest.json" --file "$DIR/manifest.json" --remote \
  --content-type application/json

STALE=$(mktemp)
python3 scripts/r2-orphans.py "$OLD" "$DIR/manifest.json" > "$STALE"
while read -r stale; do
  [ -n "$stale" ] || continue
  echo "==> pruning orphan $stale"
  if npx wrangler r2 object delete "$BUCKET/$stale" --remote >/dev/null 2>&1; then
    echo "    deleted"
  else
    echo "    could not delete -- remove it by hand"
  fi
done < "$STALE"
rm -f "$OLD" "$STALE"

echo
echo "uploaded. verify:  curl -s https://quivermutationdb.org/api/bulk | head -c 400"
