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
- **Schema v2** (`src/db/schema.ts`, migrations in `drizzle/`; design in
  `docs/PHASE2.md`): skinny browse tables `quivers` and `mutation_classes`
  with composite `(n, sortcol, id)` indexes matching the queries; **one row
  per labeled matrix in `labelings`** and per frontier matrix in
  `frontier_quivers` (D1 caps a row at 2 MB — orbits are never JSON blobs);
  `mutation_classes.exploration` ∈ complete | bound | truncated; curated
  `class_nicknames`; ingest-time aggregates + provenance in `rank_stats`
  (homepage stats and `/random/*` come from there — never scans, never
  `ORDER BY RANDOM()`); `downloads` logs exports.
- **Every list is keyset-paged** (`next_cursor` / `?cursor=`, `src/api/cursor.ts`);
  class members are served by `/classes/{id}/quivers` and `/classes/{id}/labelings`;
  the class detail embeds only the first member page and inlines labelings only
  for classes ≤ 200 matrices. Never load a whole orbit in the Worker (128 MB).
- **Agents**: `/mcp` (stateless MCP server, `src/mcp.ts`, tools wrap the API
  functions), `/api/openapi.json` (`src/api/openapi.ts`, keep in step with
  routes), `/api/export.ndjson` (resumable bulk pull), `public/llms.txt`.
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
python scripts/populate.py --count-only --max-vertices 10 --bound 2   # exact cell sizes first!
python scripts/populate.py --export-d1 dist/d1 --max-vertices 5 --bound 2 --node-cap 500 --workers 8
python scripts/populate.py --export-d1 dist/d1 --ranks 7 --bound 2 --generator sample --sample 200000 --node-cap 200
scripts/import-d1.sh dist/d1 --remote          # parts in order, ranks ascending
```

Seeds come from `qmd/census.py`: **orderly generation** (exact census of the
cell (n, bound); parallel) or **sampling** for cells that are not finite jobs
(see the size table in docs/PHASE2.md §1 — anything ≳ 10⁷ classes). Parallel
runs are bit-identical to serial ones. Never raise `--node-cap` above what
D1 can hold: labelings rows ≈ classes × cap.

A rank is exported as ordered parts `qmd-n{k}.001.sql, .002.sql, …` (statements
cut at 90 KB, parts at 64 MB — D1 limits); part 001 deletes the rank first, so
a rank is idempotent as a whole but must be imported part-by-part in order.
`manifest.json` records every part's sha256 and the sha256 of each lower-rank
`acyclicity-n{j}.json` checkpoint the rank consumed, so regenerating rank j
invalidates every rank above it. `--node-cap` stops a class BFS after C
labeled matrices: such classes are stored `exploration = 'truncated'` with
**unknown** finiteness (never finite). `docs/SCALING.md` is the original
audit; `docs/PHASE2.md` §1 is the generation-scope decision still open.

## Curated nicknames

`data/nicknames.json` is the source of truth (never the database). Each entry
names a class by `mc_id` **and** a member `matrix`, so it survives regeneration
and can be re-resolved after a re-keying. Adding one = edit the file, merge to
`main` (that is the access control). `python scripts/nicknames.py --check`
validates (CI runs it); `--sql dist/nicknames.sql` renders the table;
`--resolve` recomputes ids from matrices. Rank imports never touch
`class_nicknames`.

## Releasing data to production

Order matters — the Worker code assumes schema v2 and the data:

```bash
scripts/release-data.sh dist/d1     # migrations → rank parts → nicknames → deploy
```

## Development

```bash
npm install && npm run cf-typegen      # deps + generate Env types
npm run db:migrate:local               # schema into local D1
python scripts/populate.py --export-d1 dist/d1   # generate dataset
scripts/import-d1.sh dist/d1           # load it (parts in order)
python scripts/nicknames.py --sql dist/nicknames.sql && npx wrangler d1 execute qmd --local --file=dist/nicknames.sql
npm run dev                            # http://127.0.0.1:8787
npm run typecheck
npm run test:api                       # ~80 API assertions against wrangler dev
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
   / `src/api/classes.ts`, `EXPORT_COLUMNS` in `src/api/export.ts` (append,
   never reorder), the OpenAPI schemas in `src/api/openapi.ts`, and the MCP
   tool descriptions in `src/mcp.ts`. Extend `scripts/api-smoke.mjs`.
5. `public/` — show it on the quiver/class page and (optionally) as a
   Browse/Search column; add a `<section id="...">` definition in
   `public/wiki.html` (its section ids are the deep-link anchors every
   property label points at).

## Human-approved actions (never automatic)

DNS changes on quivermutationdb.org, deleting or suspending external
infrastructure, and archiving repositories. When a product decision isn't
covered here, ask the maintainer rather than assume:
Blake Jackson (jackson@icarm.io).
