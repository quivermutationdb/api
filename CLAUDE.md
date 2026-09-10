# QMD — Quiver Mutation Database

Backend + frontend of https://www.quivermutationdb.org — a curated, citable
database of quivers, exchange matrices, and mutation-equivalence classes.
This is research infrastructure: treat the public API as versioned-by-
politeness (keep response shapes stable), and keep CC-BY-4.0 attribution
intact.

The whole system runs on Cloudflare: one Worker serving the site and the API,
over **one PlanetScale Postgres database reached through Hyperdrive**. Earlier
stacks (Neon + Render, then D1) are decommissioned and survive only in git
history. This file describes the current system.

> **Moved off D1 on 2026-09-09.** The API serves PlanetScale Postgres (PS-40,
> provisioned through the Cloudflare dashboard so it bills to the ICARM
> Cloudflare account) via the Hyperdrive binding. All 50,828,164 rows across
> ranks 1-8 are loaded, indexed and verified. The five D1 databases still
> exist, unreferenced, as the rollback; deleting them is a human-approved
> action. Record of the migration, the measurements behind it and the load
> defects it exposed: `docs/PLANETSCALE.md`.

## Architecture

- **One Cloudflare Worker** (`qmd`, wrangler.jsonc) serves both the API
  (mounted at `/api/*`; Hono + Drizzle over Postgres) and the static frontend
  (Workers Static Assets from `public/`). Same origin, no CORS. Production
  hostnames: quivermutationdb.org + www (Custom Domains, declared in
  wrangler.jsonc `routes`).
- **One Postgres database** (PlanetScale, bound as `HYPERDRIVE`). All DB
  access goes through the routing seam `shardFor(n)` in `src/db/shard.ts` —
  today a one-liner returning the single handle; a future per-`n` split
  changes only that module. IDs encode the rank (`Q.n4.{sha256[:16]}`,
  `MC.n4.{sha256[:16]}`), so a malformed id is rejected before a query.
- **One `pg` Pool per request**, on a shallow copy of `env` (never a mutation:
  `env` is shared across an isolate), closed with `ctx.waitUntil` so a streamed
  export is not cut off. A Pool and not a Client: `pg`'s Client allows exactly
  one query at a time and the list handlers deliberately run a count alongside
  a page read. PS-40 allows **25 connections**; exhaustion surfaces as the
  typed `Unavailable` → 503 with `Retry-After`, never an untyped 500.
- **Schema v3** (`src/db/schema.ts`, migrations in `drizzle/`; design in
  `docs/PHASE2.md` + `docs/PHASE3.md`): skinny browse tables `quivers` and
  `mutation_classes`; matrices stored in the compact upper-triangular
  encoding (`qmd/encoding.py` ↔ `src/db/matrix.ts`); **`seq`**, an integer
  assigned at load by `row_number() OVER (PARTITION BY n ORDER BY id)` and
  verified gapless 1..count per rank, so `(n, seq)` is id order — it is the
  keyset tiebreak and the random-pick key, replacing SQLite's rowid; per-quiver `mutation_finite` (three-state, known even
  without a class row via Derksen–Owen) and `mutation_class_id` NULL for
  unexplored quivers; **one row per labeled matrix in `labelings`, stored only
  for complete classes with distinct × n! ≤ 200k** (`class_size` NULL
  otherwise); `mutation_classes.exploration` ∈ complete | bound | truncated;
  curated `class_nicknames`; ingest-time aggregates + provenance in
  `rank_stats`; `downloads` logs exports.
- **Two schema files, and the serving one is `src/db/schema.ts`** (Postgres;
  DDL is hand-written in `drizzle-pg/*.sql`, which is authoritative and
  deliberately withholds primary keys so the bulk load hits a bare heap).
  `src/db/schema.sqlite.ts` + `drizzle/` describe the offline pipeline's
  *intermediate* SQLite only. Shared domain types live in `src/db/types.ts` so
  neither depends on the other. Never point `drizzle.config.ts` at the
  Postgres schema.
- **`data/shards.json` is pipeline-only.** It still names the D1-format parts
  the exporter writes (`qmd/d1_export.py`, `qmd/bigcell.py`); the Worker does
  not read it. Deleting it breaks generation.
- **A rank's quiver rows are the cell plus the mutation-finite classes.**
  Exploration always runs at `EXPLORE_BOUND = 2`, so a cell taken at a lower
  bound (ranks 7-8 use `|b_ij| <= 1`) reaches weight-2 quivers outside it;
  `build_rank_rows` drops those unless they belong to a completely explored
  class. Never store the overflow — at rank 7 it is tens of millions of
  arbitrary rows against a 2.12 M cell (docs/PHASE3.md).
- **The census is of connected quivers only.** Class discovery is an
  *unlabeled* BFS (`qmd/core._bfs_unlabeled`, canonicalise every mutation
  result); the exploration bound is the constant `EXPLORE_BOUND = 2` (the
  wall at 3 is the Derksen–Owen witness); a wall crossing beats the node cap.
  Never explore labeled orbits at rank ≥ 6 (E8's is ~6×10⁷ matrices).
- **Every list is keyset-paged** (`next_cursor` / `?cursor=`, `src/api/cursor.ts`);
  class members are served by `/classes/{id}/quivers` and `/classes/{id}/labelings`;
  the class detail embeds only the first member page and inlines labelings only
  for classes ≤ 200 matrices. Never load a whole orbit in the Worker (128 MB).
- **Agents**: `/mcp` (stateless MCP server, `src/mcp.ts`, tools wrap the API
  functions), `/api/openapi.json` (`src/api/openapi.ts`, keep in step with
  routes), `/api/export.ndjson` (resumable bulk pull), `public/llms.txt`,
  `/api/lookup` (paste a matrix → canonical id → row; `src/canon.ts` is the
  TS port of the lex-min definition and must stay identical to Python's).
- **The Python math pipeline stays Python and stays offline.** `qmd/core.py`
  (mutation, canonical hashing, BFS generation) never runs on Cloudflare.
  Keep the Worker lean: no heavy computation, no matrix math server-side —
  e.g. a quiver's canonical matrix is *looked up* from the quivers table, not
  recomputed. Rows are tiny; when in doubt, prefer an extra indexed column
  over a query-time computation.
- **Excel export is generated client-side** from CSV (`public/xlsx-lite.js`);
  the Worker serves CSV only (`/api/export`, streamed from paginated reads).

## Data pipeline (offline → Postgres)

```bash
python scripts/populate.py --count-only --max-vertices 10 --bound 2   # exact (connected) cell sizes first!
python scripts/populate.py --export-d1 dist/d1 --ranks 4 --bound 10 --node-cap 100 --workers 8
python scripts/populate.py --export-d1 dist/d1 --ranks 8 --bound 1 --generator sample --sample 1000000 --node-cap 100 --workers 8
scripts/run-rank6.sh   # rank 6 via qmd/bigcell.py; resumable, detached, native arm64 python
scripts/release-data.sh dist/d1                # parts -> SQLite -> COPY -> Postgres -> deploy
```

The agreed cells and the cost model are in docs/PHASE3.md §1/§3. Seeds
with an entry |b_ij| ≥ 3 are marked mutation-infinite without exploration;
rank 6 runs through the streaming pipeline (`qmd/bigcell.py`, scratch
SQLite, resumable stages) with class rows only for a sample. **In that
pipeline, never update the big scratch table row by row** — a scattered
`WHERE id = ?` costs one random page read per row and stalls the job for
weeks; stage verdicts in a side table and fold them in one id-ordered pass,
and resume from a committed watermark rather than scanning for unfinished
rows (docs/PHASE3.md).

**The capped label pass cannot decide a class bigger than its cap**, so every
mutation-finite class above it comes back *unknown* — and a uniform sample
will not rescue them (a 49-quiver class in a 42.5 M cell is a 1-in-a-million
target; rank 6 caught A6/D6/E6 by luck and missed ten other finite classes
entirely). Always follow the label pass with `stage_resolve` (re-explore the
leftovers at a cap ~1000×, which costs seconds) and explore the
mutation-finite classes explicitly with `stage_finite_classes` rather than
hoping the sample finds them. Mutation-finite quivers are rare — 428 of
42.5 M at rank 6, in 13 classes — and they are the mathematically
interesting ones, so they must never be left to chance.

**Never name a mutation class by hand.** `qmd/surfaces.py` builds marked
surfaces from triangle gluings, reads off `(genus, boundary, punctures)` by
Euler's formula and the adjacency quiver from the same data, and so generates
`mc_id -> surface name` for every surface class of a rank (`reference_for`,
cached in `dist/surface-reference.json`; guarded by `SURFACE_MAX_RANK`, 8 by
default). `d1_export` fills `label` from `dynkin.classify` and falls back to
`surfaces.classify`, so a new rank names itself. Anything a rank leaves
unlabelled is genuinely exceptional — at rank 6 exactly X6 — and only those
need an entry in `data/nicknames.json`.

Seeds come from `qmd/census.py`: **orderly generation** (exact census of the
cell (n, bound); parallel) or **sampling** for cells that are not finite jobs
(see the size table in docs/PHASE2.md §1 — anything ≳ 10⁷ classes). Parallel
runs are bit-identical to serial ones. `--node-cap` bounds the labelings
table: rows ≈ classes × cap.

A rank is exported as ordered parts `qmd-n{k}.001.sql, .002.sql, …` (statements
cut at 90 KB, parts at 64 MB). These sizes were D1's limits and are now just
the intermediate format's: `scripts/pg-load-local.sh` and `release-data.sh`
replay the parts into one SQLite, then `scripts/pg-export.py` renders Postgres
COPY text. Part 001 deletes the rank first, so a rank is idempotent as a whole
but must be replayed part-by-part in order.
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

Order matters — the Worker is deployed LAST, after the data it expects:

```bash
set -a && . ./.pgenv && set +a      # Postgres connection (NOT .env -- see below)
set -a && . ./.env   && set +a      # CLOUDFLARE_API_TOKEN, for the deploy
scripts/release-data.sh dist/d1     # verify → schema → data → indexes →
                                    # patches+nicknames → verify → deploy
```

`release-data.sh` is a FULL REBUILD (`0001_init.sql` starts from nothing), not
an incremental update. Adding one rank to a live database is a different job:
follow docs/PLANETSCALE.md, and chunk anything the size of rank 6.

## Development

```bash
npm install && npm run cf-typegen      # deps + generate Env types
createdb qmd                           # a local Postgres 17+ to develop against
npm run db:load:local                  # generate the dev cell + load it (one step)
npm run db:verify                      # structural checks on what just loaded
npm run dev                            # http://127.0.0.1:8787
npm run typecheck
npm run test:api                       # ~80 API assertions against wrangler dev
npx playwright install chromium        # once; then:
npm run test:browser                   # Chromium end-to-end page checks
python -m pytest tests/ -q             # math pipeline suite (incl. golden IDs)
npm run deploy                         # production (needs CLOUDFLARE_API_TOKEN)
```

CI (`.github/workflows/ci.yml`) runs pytest, typecheck, and the API smoke
tests against a Postgres service container loaded by `pg-load-local.sh` — the
same script and the same load path a release uses, so CI cannot pass against a
path production does not take.

**Connection settings live in `.pgenv`, not `.env`.** Wrangler auto-loads
`.env` into the Worker's `process.env`, where the `pg` driver reads
`PGSSLNEGOTIATION`/`PGSSLMODE` and fails with "sslnegotiation=direct requires
SSL to be enabled". `.env` keeps `CLOUDFLARE_API_TOKEN` only. `wrangler dev`
reaches the database through the Hyperdrive binding's `localConnectionString`
(or `CLOUDFLARE_HYPERDRIVE_LOCAL_CONNECTION_STRING_HYPERDRIVE`), which
**must include a password** — Miniflare rejects a passwordless one even when
the server uses trust auth.

**IDs are frozen.** `tests/golden/ids-n4.json` pins every published
`Q.*`/`MC.*` id (and class membership/sizes). A change that re-keys the
database is a breaking change for citations; if it is ever intended, regenerate
the golden file deliberately (`python tests/test_golden_ids.py --regenerate`)
and ship an alias table for the old ids.

Use a **scoped API token** (Workers edit on this account only); never a global
key — this is a shared organizational Cloudflare account (ICARM). The token is
deliberately too narrow to read account membership or Hyperdrive configs; that
is working as intended, not something to widen.

## Known defect: dangling class references at rank 8

**Read this before regenerating any rank, or adding rank 9.**

1,749 of the 258,033 rank-8 quivers (0.68 %) carry a `mutation_class_id`
naming a `mutation_classes` row that was never written — 1,740 distinct ids,
all mutation-**infinite**, all in-cell (`max_edge = 1`), `labeling_count` NULL.
Left alone, those quivers report `explored: true` and link to a class page that
404s.

The database is patched: `drizzle-pg/0003_rank8_dangling_class_refs.sql` NULLs
the reference (NULL already means "not explored", so the rows land in an
existing honest state), is idempotent, and runs as step 5 of
`scripts/release-data.sh`. `mutation_finite` is deliberately left alone — it is
a separate column with its own provenance, and the infiniteness verdict came
from a real exploration that crossed the weight bound. Derksen–Owen does *not*
cover these; they are `max_edge = 1`.

**The exporter-side cause is UNRESOLVED.** The same 1,749 are present in the
source SQLite and in `dist/d1`, where a sampled id
(`MC.n8.000d1755ebbc3f9c`) appears exactly once — as a quiver's class
reference, never as a class row. Rank 8 was generated by an earlier state of
`qmd/d1_export.py`, so reading today's exporter does not explain it, and no
filter drops class rows at render time (`rank_stats.class_count` reconciles
with what was written). Do not claim it is fixed on the strength of a code
reading.

**The test is behavioural:** regenerate rank 8, reload, and run `0003`. If it
reports `UPDATE 0`, the defect is gone. If it still patches rows, the bug is
live in the current exporter and a new rank will inherit it — investigate
`build_rank_rows` / `render_rank_sql` in `qmd/d1_export.py` before publishing.
`scripts/pg-verify.sql` asserts the count is 0 after the patch, so a release
cannot ship the broken state silently.

## Adding a new invariant / property

Keep these in sync:

1. **Both schema files, and the hand-written DDL.**
   `src/db/schema.ts` (Postgres, what the Worker serves) *and*
   `drizzle-pg/0001_init.sql` (authoritative DDL — add the column there by
   hand; `drizzle-kit` must never be pointed at the Postgres schema), plus an
   index in `drizzle-pg/0002_indexes.sql` if it is filterable. Then
   `src/db/schema.sqlite.ts` + `npm run db:generate:sqlite`, because the
   offline pipeline's intermediate carries the column too. A type shared by
   both belongs in `src/db/types.ts`.
2. `qmd/invariants.py` or `qmd/local_acyclicity.py` — compute it.
3. `qmd/d1_export.py` — write it (build_rank_rows + the column list in
   render_rank_sql) *and* `scripts/pg-export.py` picks the column list up from
   there, so it needs no edit; regenerate and reload the dataset.
4. Worker API — surface it: list/detail serializers in `src/api/quivers.ts`
   / `src/api/classes.ts`, `EXPORT_COLUMNS` in `src/api/export.ts` (append,
   never reorder), the OpenAPI schemas in `src/api/openapi.ts`, and the MCP
   tool descriptions in `src/mcp.ts`. Extend `scripts/api-smoke.mjs`. If it is
   filterable, benchmark the filtered count on rank 6 before shipping: an
   unindexed filter there scans a 9 GB heap (see `TOTAL_CAP` in
   `src/api/quivers.ts` and docs/PLANETSCALE.md).
5. `public/` — show it on the quiver/class page and (optionally) as a
   Browse/Search column; add a `<section id="...">` definition in
   `public/wiki.html` (its section ids are the deep-link anchors every
   property label points at).

## Human-approved actions (never automatic)

DNS changes on quivermutationdb.org, deleting or suspending external
infrastructure, and archiving repositories. When a product decision isn't
covered here, ask the maintainer rather than assume:
Blake Jackson (jackson@icarm.io).
