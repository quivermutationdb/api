# QMD API

Backend for the [Quiver Mutation Database](https://quivermutationdb.org).

> **Migration in progress:** the backend is moving from Neon (Postgres) +
> Render (FastAPI) to Cloudflare (Workers + D1). See `CLAUDE.md` for the
> migration brief. The Python math pipeline (`qmd/`) stays Python and runs
> offline; only serving moves to the Worker.

## Cloudflare Worker

One Worker serves the API (`/api/*`, Hono + Drizzle over D1) and the static
frontend (Workers Static Assets from `public/`).

```
src/
├── index.ts         # Worker entry point (Hono app + assets fallthrough)
├── api/index.ts     # API routes, mounted at /api
└── db/
    ├── schema.ts    # Drizzle schema (D1/SQLite mirror of the Postgres schema)
    └── shard.ts     # shardFor(n) — the single DB routing seam
drizzle/             # Generated SQL migrations (wrangler d1 migrations apply)
public/              # Static frontend (placeholder until the website repo is folded in)
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
api/
├── qmd/
│   ├── core.py          # Matrix types, mutation, ID generation, BFS explorer
│   └── __init__.py
├── app/                 # FastAPI app (coming soon)
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   └── routers/
├── data/
│   └── seeds/           # Existing seed data (JSON)
├── scripts/
│   └── ingest.py        # Load seed data into PostgreSQL
├── tests/
│   └── test_core.py     # Full test suite for core.py
├── requirements.txt
└── README.md
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
pip install -r requirements.txt

# Run the generation pipeline
python -c "
from qmd.core import run_generation
r = run_generation(max_vertices=4, bound=2)
print(f'{len(r.quivers)} quivers in {len(r.classes)} mutation classes')
"

# Run tests
python -m pytest tests/ -v
```

## Known Results (sanity checks)

| Quiver type | Mutation class size |
|---|---|
| A2          | 2                   |
| A3          | 14                  |
| D4          | 132 (finite type)   |

## Contact

Maintained by Blake Jackson — <jackson@icarm.io>

## License

CC-BY-4.0
