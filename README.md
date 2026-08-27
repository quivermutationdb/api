# QMD API

Backend for the [Quiver Mutation Database](https://quivermutationdb.org).

One Cloudflare Worker serves the site, the JSON API (`/api/*`) and an MCP
server for agents (`/mcp`), backed by a D1 database. The Python math pipeline
(`qmd/`) runs offline and exports SQL for D1. See `CLAUDE.md` for the
architecture guide and `docs/PHASE2.md` for the scaling design.

**For agents and scripts:** [`/llms.txt`](https://quivermutationdb.org/llms.txt) ·
[`/api/openapi.json`](https://quivermutationdb.org/api/openapi.json) ·
MCP at `https://quivermutationdb.org/mcp` · bulk pulls via
`/api/export.ndjson` (follow `X-Next-Cursor`). Every list returns `next_cursor`.

## Cloudflare Worker

One Worker serves the API (`/api/*`, Hono + Drizzle over D1) and the static
frontend (Workers Static Assets from `public/`).

```
src/
├── index.ts         # Worker entry point (Hono app, /mcp, assets fallthrough)
├── mcp.ts           # Model Context Protocol server (tools over the API functions)
├── api/index.ts     # API routes, mounted at /api (cursor.ts, openapi.ts, ...)
└── db/
    ├── schema.ts    # Drizzle schema v2 for D1 (labelings rows, nicknames, ...)
    └── shard.ts     # shardFor(n) — the single DB routing seam
drizzle/             # SQL migrations (wrangler d1 migrations apply)
data/nicknames.json  # Curated class nicknames (source of truth)
public/              # Static frontend, served as Workers Static Assets
wrangler.jsonc       # Worker + D1 + Static Assets config
```

```bash
npm install
npm run cf-typegen         # generate worker-configuration.d.ts (Env types)
npm run db:migrate:local   # apply migrations to the local D1 database
npm run dev                # wrangler dev → http://127.0.0.1:8787
npm run typecheck
```

### Loading data

The Python pipeline exports each rank as ordered SQL parts (resumable; re-runs
skip up-to-date ranks — see `qmd/d1_export.py`):

```bash
python scripts/populate.py --export-d1 dist/d1   # dist/d1/qmd-n{k}.NNN.sql + manifest
scripts/import-d1.sh dist/d1                     # --remote for production
python scripts/nicknames.py --sql dist/nicknames.sql && npx wrangler d1 execute qmd --local --file=dist/nicknames.sql
```

Production release (migrations → data → nicknames → deploy, in that order):
`scripts/release-data.sh dist/d1`.

## Structure

```
qmd/                     # Offline math pipeline (pure Python, stdlib only)
├── core.py              # Matrix types, mutation, ID generation, BFS explorer
├── canonicalize.py      # Canonical forms
├── invariants.py        # Per-quiver invariants
├── local_acyclicity.py  # Banff / Louise / p-prime searches
├── dynkin.py            # Dynkin classification
├── class_properties.py  # Per-class property resolution (shared logic)
└── d1_export.py         # GenerationResult -> multipart per-rank SQL for D1
scripts/
├── populate.py          # Generate + export the dataset (--node-cap, --bound)
├── import-d1.sh         # Import parts in the correct order (local/remote)
├── nicknames.py         # Validate / render / re-resolve data/nicknames.json
├── release-data.sh      # Production release in the safe order
├── api-smoke.mjs        # API assertions against wrangler dev
└── browser-check.mjs    # Chromium end-to-end page checks
tests/
└── test_core.py         # Math pipeline test suite
```

## Identifiers

**Quiver ID:** `Q.n{vertices}.{sha256[:16]}`
- Hashes the labeled exchange matrix (row-major JSON, compact)
- Example: `Q.n4.a3f2c1d9e8b70f21`

**Mutation Class ID:** `MC.n{vertices}.{sha256[:16]}`
- Hashes the lex-min exchange matrix across all bounded-mutation-reachable
  matrices in the class (no vertex relabeling)
- Example: `MC.n4.f8a21c3d7e904b56`

## Generation Rules

- Seed quivers: all skew-symmetric `{0, ±1, ±2}` matrices on ≤ 4 vertices
- Mutation bound: `|b_ij| ≤ 2` at every step
- If a mutation would produce `|b_ij| > 2`, that branch is stopped and the
  class is marked `is_open = True`

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # pytest only; the pipeline is stdlib-pure

# Run the generation pipeline
python -c "
from qmd.core import run_generation
r = run_generation(max_vertices=4, bound=2)
print(f'{len(r.quivers)} quivers in {len(r.classes)} mutation classes')
"

# Run tests (the golden-ID test guards every published Q.*/MC.* id)
python -m pytest tests/ -q
```

## Known Results (sanity checks)

| Quiver type | Labeled exchange matrices in the class | Distinct quivers |
|---|---|---|
| A2          | 2   | 1 |
| A3          | 14  | 4 |
| D4          | 50  | 6 |

## Contact

Maintained by Blake Jackson — <jackson@icarm.io>

## License

CC-BY-4.0
