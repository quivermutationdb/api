# QMD — Cloudflare Migration Brief

This repo is the backend of the **Quiver Mutation Database** (https://www.quivermutationdb.org), a curated, citable database of quivers, exchange matrices, and mutation-equivalence classes. We are migrating the backend from **Neon (Postgres) + Render (FastAPI)** to **Cloudflare (Workers + D1)**. This document is the authoritative spec: follow it before improvising, and prefer current Cloudflare docs (via the Cloudflare skill / docs MCP) over pre-trained knowledge for any tooling or config detail.

## Target architecture (settled — do not relitigate)

- **One Cloudflare Worker** serves both the API (mounted at `/api/*`) and the static frontend (Workers Static Assets). Same origin, no CORS.
- **API in TypeScript**: Hono for routing, Drizzle ORM with the D1 adapter, Drizzle migrations (replacing Alembic).
- **One D1 database** for now (data is small: ranks 1–4). Design for later **per-`n` sharding**:
  - All DB access goes through a single routing seam, e.g. `shardFor(n: number): D1Database`, which today returns the one bound DB for every `n`.
  - IDs already encode `n` (`Q.n4.{sha256[:16]}`, `MC.n4.{sha256[:16]}`), so point lookups can always be routed by parsing the ID prefix.
  - Keep search/browse tables **skinny** (IDs + invariant columns only, matrices referenced not inlined) so they can later serve as a global index DB while matrix payloads move into per-`n` shards.
- **The Python math pipeline stays Python and stays offline.** `qmd/core.py` (mutation, canonical hashing, BFS generation) never runs on Cloudflare. `scripts/ingest.py` gains an export mode that emits SQL for `wrangler d1 import` (or hits the D1 REST API) instead of writing to Postgres.
- **Generation will run on cloud compute** (full-database computation exceeds the maintainer's laptop), so the export mode must produce self-contained artifacts, not assume a local wrangler session: emit **one SQL/SQLite file per `n`** (this pre-aligns ingest with the future per-`n` shards), make generation resumable/checkpointable, and treat the per-`n` files as the hand-off between the compute environment and `wrangler d1 import` (which can run from the cloud box with the scoped API token, or from the laptop after downloading the artifacts).
- **No Postgres, no Hyperdrive, no Containers** in the target state. Neon and Render are decommissioned at the end (human-approved step).

## Data model (mirror the existing Postgres schema; confirm against `alembic/` and `scripts/ingest.py`)

Core entities, with indexes on every filterable/sortable column:

- `quivers`: `id` (PK, `Q.n{k}.{hash}`), `n` (int), `exchange_matrix` (JSON text, row-major), `mutation_class_id` (FK), per-quiver invariants.
- `mutation_classes`: `id` (PK, `MC.n{k}.{hash}`), `n`, `class_size`, `is_open` (bool — mutation bound `|b_ij| ≤ 2` exceeded), invariants: `mutation_finite`, `acyclic`, `dynkin_type`, `representation_type`, `connectivity` — each with **provenance** fields (values on the site carry explicit provenance).
- Aggregate/stats table(s) written at ingest time (distinct quiver count, labeled quiver count, per-rank counts) to serve the homepage stats without scans.

Default sort order everywhere: `n` first, then ID. This makes cross-`n` pagination trivial (exhaust one `n`, cursor into the next) and matches the natural presentation order.

## API endpoints (drive from what the live frontend calls; verify against the deployed site's network requests)

- `GET /api/stats` — homepage counts (from the aggregates table; edge-cacheable).
- `GET /api/quivers` and `GET /api/classes` — browse: paginated, sortable, filterable by rank, Dynkin type, representation type, connectivity, `mutation_finite`, `acyclic`, `is_open`, class size ranges.
- `GET /api/quivers/{id}`, `GET /api/classes/{id}` — detail (route by ID prefix through `shardFor`).
- `GET /api/random/quiver`, `GET /api/random/class` — via stored counts + offset (no `ORDER BY RANDOM()` scans).
- `GET /api/export.csv` — CSV export of any filtered cut, streamed from paginated reads. Excel export is generated client-side from CSV; do not build xlsx generation into the Worker.
- Preserve existing response shapes where the frontend already expects them; otherwise keep JSON flat and stable (this is citable research infrastructure — treat the API as versioned-by-politeness).

## Migration sequence

1. Scaffold the Worker project in this repo: `wrangler.jsonc`, Hono app, Drizzle schema + migrations, Static Assets config. TypeScript strict.
2. Create the D1 database (name: `qmd`) in the organization's Cloudflare account; bind as `DB`.
3. Add the export mode to `scripts/ingest.py` (Postgres writes → SQL file for `wrangler d1 import`). Regenerating from seeds and loading fresh is acceptable at current data size; a Neon dump is not required.
4. Implement endpoints; test with `wrangler dev` against a locally imported dataset; port relevant tests from `tests/`.
5. Fold the frontend into this repo: copy the static site from https://github.com/quivermutationdb/website into this project's assets directory (preserving its git history via `git subtree` if convenient, plain copy if not), switch its API base URL to same-origin `/api`, and leave a README pointer in the old `website` repo before archiving it (archiving is a human-approved step).
6. Deploy to a workers.dev preview URL first. **Human-approved steps, never automatic:** pointing `quivermutationdb.org` DNS at the Worker, and decommissioning Render/Neon.
7. Later (not now): enable D1 read replication via the Sessions API when traffic justifies it; introduce real per-`n` shards behind `shardFor` when any table's growth curve approaches D1's per-database limit.

## Constraints and conventions

- Use a **scoped API token** (Workers + D1 edit on this account only); never a global key. This is a shared organizational account.
- License is CC-BY-4.0; keep attribution intact.
- Keep the Worker lean: no heavy computation, no matrix math server-side.
- Rows are tiny; when in doubt, prefer an extra indexed column over a query-time computation.
- Maintainer: Blake Jackson (jackson@icarm.io). When a product decision isn't covered here, ask rather than assume.
