# QMD — Quiver Mutation Database

Backend + frontend of https://www.quivermutationdb.org — a curated, citable
database of quivers, exchange matrices, and mutation-equivalence classes.
This is research infrastructure: treat the public API as versioned-by-
politeness (keep response shapes stable), and keep CC-BY-4.0 attribution
intact.

The whole system runs on Cloudflare (one Worker + one D1 database); the
earlier hosting stack was decommissioned in August 2026 and survives only in
git history. This file describes the current system.

## Architecture

- **One Cloudflare Worker** (`qmd`, wrangler.jsonc) serves both the API
  (mounted at `/api/*`; Hono + Drizzle over D1) and the static frontend
  (Workers Static Assets from `public/`). Same origin, no CORS. Production
  hostnames: quivermutationdb.org + www (Custom Domains, declared in
  wrangler.jsonc `routes`).
- **One D1 database** (`qmd`, bound as `DB`). All DB access goes through the
  routing seam `shardFor(n)` in `src/db/shard.ts` — today it returns the one
  bound DB; future per-`n` shards change only that module. IDs encode the
  rank (`Q.n4.{sha256[:16]}`, `MC.n4.{sha256[:16]}`), so point lookups route
  by prefix.
- **Schema** (`src/db/schema.ts`, migrations in `drizzle/`): skinny browse
  tables `quivers` and `mutation_classes` (every filterable column indexed,
  composite `(n, id)` for the default sort — `n` first, then id); heavy orbit
  JSON in `mutation_class_payloads`; ingest-time aggregates in `rank_stats`
  (homepage stats and `/random/*` come from there — never scans, never
  `ORDER BY RANDOM()`); `downloads` logs exports.
- **The Python math pipeline stays Python and stays offline.** `qmd/core.py`
  (mutation, canonical hashing, BFS generation) never runs on Cloudflare.
  Keep the Worker lean: no heavy computation, no matrix math server-side —
  e.g. a quiver's canonical matrix is *looked up* from the quivers table, not
  recomputed. Rows are tiny; when in doubt, prefer an extra indexed column
  over a query-time computation.
- **Excel export is generated client-side** from CSV (`public/xlsx-lite.js`);
  the Worker serves CSV only (`/api/export`, streamed from paginated reads).

## Data pipeline (offline → D1)

```bash
python scripts/populate.py --export-d1 dist/d1        # one SQL file per rank
npx wrangler d1 execute qmd --remote --file=dist/d1/qmd-n4.sql   # per rank
```

Per-rank files are self-contained and idempotent (each replaces its rank's
rows, including the rank_stats row). Generation is resumable: manifest.json
tracks file hashes, and acyclicity-n{k}.json checkpoints feed the
mutation-acyclicity subquiver fallback (which consumes lower-rank results —
ranks must be produced in ascending order). See `qmd/d1_export.py`. This is
designed to run on cloud compute for the future larger-rank generation
(docs/SCALING.md is the phase-2 planning doc).

## Development

```bash
npm install && npm run cf-typegen      # deps + generate Env types
npm run db:migrate:local               # schema into local D1
python scripts/populate.py --export-d1 dist/d1   # generate dataset
for f in dist/d1/qmd-n*.sql; do npx wrangler d1 execute qmd --local --file=$f; done
npm run dev                            # http://127.0.0.1:8787
npm run typecheck
npm run test:api                       # ~50 API assertions against wrangler dev
npx playwright install chromium        # once; then:
npm run test:browser                   # Chromium end-to-end page checks
python -m pytest tests/ -q             # math pipeline suite (incl. golden IDs)
npm run deploy                         # production (needs CLOUDFLARE_API_TOKEN)
```

CI (`.github/workflows/ci.yml`) runs pytest, typecheck, and the API smoke
tests against a freshly generated local D1 on every push and PR.

**IDs are frozen.** `tests/golden/ids-n4.json` pins every published
`Q.*`/`MC.*` id (and class membership/sizes). A change that re-keys the
database is a breaking change for citations; if it is ever intended, regenerate
the golden file deliberately (`python tests/test_golden_ids.py --regenerate`)
and ship an alias table for the old ids.

Use a **scoped API token** (Workers + D1 edit on this account only); never a
global key — this is a shared organizational Cloudflare account (ICARM).

## Adding a new invariant / property

Keep these in sync:

1. `src/db/schema.ts` — add the column (+ index if filterable), then
   `npm run db:generate` and apply the migration locally and remotely.
2. `qmd/invariants.py` or `qmd/local_acyclicity.py` — compute it.
3. `qmd/d1_export.py` — write it (build_rank_rows + the column list in
   render_rank_sql); regenerate and re-import the dataset.
4. Worker API — surface it: list/detail serializers in `src/api/quivers.ts`
   / `src/api/classes.ts`, and `EXPORT_COLUMNS` in `src/api/export.ts`.
   Extend `scripts/api-smoke.mjs` to cover it.
5. `public/` — show it on the quiver/class page and (optionally) as a
   Browse/Search column; add a `<section id="...">` definition in
   `public/wiki.html` (its section ids are the deep-link anchors every
   property label points at).

## Human-approved actions (never automatic)

DNS changes on quivermutationdb.org, deleting or suspending external
infrastructure, and archiving repositories. When a product decision isn't
covered here, ask the maintainer rather than assume:
Blake Jackson (jackson@icarm.io).
