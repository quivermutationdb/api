# QMD API

Backend for the [Quiver Mutation Database](https://quivermutationdb.org).

One Cloudflare Worker serves the site and the API (same origin, `/api/*`),
backed by a D1 database. The Python math pipeline (`qmd/`) runs offline and
exports SQL for D1. See `CLAUDE.md` for the architecture guide.

## Cloudflare Worker

One Worker serves the API (`/api/*`, Hono + Drizzle over D1) and the static
frontend (Workers Static Assets from `public/`).

```
src/
├── index.ts         # Worker entry point (Hono app + assets fallthrough)
├── api/index.ts     # API routes, mounted at /api
└── db/
    ├── schema.ts    # Drizzle schema for D1 (SQLite)
    └── shard.ts     # shardFor(n) — the single DB routing seam
drizzle/             # Generated SQL migrations (wrangler d1 migrations apply)
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

The Python pipeline exports one self-contained SQL file per rank (resumable;
re-runs skip up-to-date ranks — see `qmd/d1_export.py`):

```bash
python scripts/populate.py --export-d1 dist/d1   # dist/d1/qmd-n{1..4}.sql
for f in dist/d1/qmd-n*.sql; do
  npx wrangler d1 execute qmd --local --file=$f   # --remote for production
done
```

## Structure

```
qmd/                     # Offline math pipeline (pure Python, stdlib only)
├── core.py              # Matrix types, mutation, ID generation, BFS explorer
├── canonicalize.py      # Canonical forms
├── invariants.py        # Per-quiver invariants
├── local_acyclicity.py  # Banff / Louise / p-prime searches
├── dynkin.py            # Dynkin classification
├── class_properties.py  # Per-class property resolution (shared logic)
└── d1_export.py         # GenerationResult -> per-rank SQL for D1
scripts/
├── populate.py          # Generate + export the dataset (one SQL file per rank)
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
