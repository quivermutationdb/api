# QMD API

Backend for the [Quiver Mutation Database](https://quivermutationdb.org).

One Cloudflare Worker serves the site, the JSON API (`/api/*`) and an MCP
server for agents (`/mcp`), backed by a D1 database. The Python math pipeline
(`qmd/`) runs offline and exports SQL for D1. See `CLAUDE.md` for the
architecture guide, `docs/PHASE2.md` and `docs/PHASE3.md` for the scaling and
census design.

> **Migration in progress:** the database is moving from D1 to PlanetScale
> Postgres, with the bulk census published to R2. Decided 2026-09-01. Both
> blockers are now cleared — ranks 7 and 8 are generated and verified, and the
> pre-flight code review is done — so the next step is provisioning. Runbook,
> measurements and open decisions: `docs/PLANETSCALE.md`.

**For agents and scripts:** [`/llms.txt`](https://quivermutationdb.org/llms.txt) ·
[`/api/openapi.json`](https://quivermutationdb.org/api/openapi.json) ·
MCP at `https://quivermutationdb.org/mcp` · bulk pulls via
`/api/export.ndjson` (follow `X-Next-Cursor`). Every list returns `next_cursor`.

## Census coverage

| rank | quiver rows | classes | mutation-finite classes | status |
| ---: | ---: | ---: | ---: | --- |
| 1–3 | 1,561 | 19 | 5 | live |
| 4 | 3,574,495 | 149 | 5 | live |
| 5 | 2,359,306 | 17,567 | 7 | live |
| 6 | 42,514,454 | 242,981 | 13 | generated, not imported |
| 7 | 2,120,315 | 35,241 | 15 | generated, not imported |
| 8 | 258,033 | 32,004 | 19 | generated, not imported |

**50,828,164 quiver rows** and **327,961** classes in total. Ranks 6–8 exist in
`dist/d1` and are verified (`scripts/verify-export.py`) but were deliberately
never loaded into D1 — the write cost would have been spent weeks before the
PlanetScale move. They ship with Postgres.

The mutation-finite counts match the Fomin–Shapiro–Thurston classification
exactly at every rank: surface types plus the exceptionals (rank 6 = 11 + E6 +
X6; rank 7 = 12 + E7 + Ẽ6 + X7; rank 8 = 16 + E8 + Ẽ7 + E6^(1,1)). Rank 8 is
sampled — its cell holds 572,849,763 connected quivers, of which 250,000 are
stored, so `census_size` there is not a row count.

## Cloudflare Worker

One Worker serves the API (`/api/*`, Hono + Drizzle over D1) and the static
frontend (Workers Static Assets from `public/`).

```
src/
├── index.ts         # Worker entry point (Hono app, /mcp, assets fallthrough)
├── mcp.ts           # Model Context Protocol server (tools over the API functions)
├── canon.ts         # Lex-min canonical form: TS port of qmd/canonicalize.py
├── api/
│   ├── index.ts     # Routes mounted at /api; error handling (400/404/503)
│   ├── quivers.ts   # Quiver list/detail, filters, sorts, totals
│   ├── classes.ts   # Class list/detail, members, labelings
│   ├── export.ts    # CSV + resumable NDJSON, streamed from paged reads
│   ├── cursor.ts    # Opaque keyset cursors (NULL placement pinned explicitly)
│   ├── merge.ts     # Cross-shard page merge
│   ├── lookup.ts    # Paste a matrix -> canonical id -> row
│   ├── random.ts    # Random quiver / class
│   ├── nicknames.ts # Curated class nicknames
│   ├── openapi.ts   # OpenAPI document (keep in step with routes)
│   └── errors.ts    # BadRequest -> 400, Unavailable -> 503
└── db/
    ├── schema.ts    # Drizzle schema v3 (quivers, classes, labelings, nicknames, stats)
    ├── shard.ts     # shardFor(n) — the single DB routing seam
    └── matrix.ts    # Upper-triangular encoding: TS port of qmd/encoding.py
drizzle/             # SQL migrations (wrangler d1 migrations apply)
data/nicknames.json  # Curated class nicknames (source of truth)
data/seeds.json      # Curated seed quivers a sampled rank must contain
data/shards.json     # Shard map (one main DB + per-rank splits)
public/              # Static frontend + wiki, served as Workers Static Assets
wrangler.jsonc       # Worker + D1 + Static Assets config
```

`src/canon.ts` and `src/db/matrix.ts` are ports of their Python counterparts and
**must stay identical** — the Q.* ids are frozen, and `/api/lookup` depends on
the TS side agreeing byte-for-byte (verified: canonical form and SHA-256 both
match Python over randomised matrices at every rank).

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

Parts must be imported **in order** — part 001 of a rank deletes that rank
first, so a rank is idempotent as a whole but not part-by-part.

Production release (migrations → data → nicknames → deploy, in that order):
`scripts/release-data.sh dist/d1`.

## Structure

```
qmd/                     # Offline math pipeline (pure Python, stdlib only)
├── core.py              # Matrix types, mutation, ID generation, BFS explorer, gluing
├── canonicalize.py      # Lex-min canonical form (branch and bound) + n! oracle
├── census.py            # Exact cell counts (Burnside) + orderly generation + sampling
├── encoding.py          # Compact upper-triangular matrix encoding
├── invariants.py        # Per-quiver invariants (Tits form, symmetry group, ...)
├── local_acyclicity.py  # Banff / Louise / P' searches (three-state)
├── dynkin.py            # Dynkin and affine classification
├── surfaces.py          # Marked surfaces: signatures, quivers, names (generated)
├── class_properties.py  # Per-class property resolution (mutation-acyclic heredity)
├── bigcell.py           # Streaming pipeline for cells too large to hold in memory
└── d1_export.py         # GenerationResult -> multipart per-rank SQL for D1
scripts/
├── populate.py          # Generate + export the dataset (--bound, --node-cap, --ranks)
├── run-rank6.sh         # Rank 6 via bigcell (resumable, detached)
├── run-rank78.sh        # Ranks 7 and 8 via bigcell — ./run-rank78.sh [7|8|both]
├── run-detached.sh      # caffeinate + detach + append-only log
├── verify-export.py     # Check exported parts against the manifest (shard-aware)
├── import-d1.sh         # Import parts in the correct order (local/remote)
├── migrate-all.sh       # Apply migrations to every shard
├── nicknames.py         # Validate / render / re-resolve data/nicknames.json
├── release-data.sh      # Production release in the safe order
├── api-smoke.mjs        # ~60 API assertions (expects wrangler dev on :8787)
└── browser-check.mjs    # Chromium end-to-end page checks
tests/                   # 128 tests: core, census, invariants, surfaces, export, golden ids
```

## Identifiers

**Quiver ID:** `Q.n{vertices}.{sha256[:16]}`
- Hashes the **row-major lex-min** matrix over all vertex relabelings
- Example: `Q.n4.a3f2c1d9e8b70f21`

**Mutation Class ID:** `MC.n{vertices}.{sha256[:16]}`
- Hashes the lex-min matrix across every bounded-mutation-reachable matrix in
  the class
- Example: `MC.n4.f8a21c3d7e904b56`

**IDs are frozen.** `tests/golden/ids-n4.json` pins every published id and class
membership. Re-keying is a breaking change for citations; regenerate it only
deliberately (`python tests/test_golden_ids.py --regenerate`) and ship an alias
table for the old ids.

## Generation

Seeds come from `qmd/census.py`, not from a fixed list:

- **Orderly generation** (canonical augmentation) enumerates a cell `(n, h)` —
  every connected quiver with `|b_ij| <= h`, each isomorphism class exactly
  once, without listing the labeled matrices. Verified against exact Burnside
  counts.
- **Sampling** for cells that are not finite jobs (rank 8's cell has 5.7×10⁸
  connected quivers).
- **Curated seeds** (`data/seeds.json`, `qmd/dynkin.py`, `qmd/surfaces.py`)
  guarantee every Dynkin, affine and surface type is present with its full
  mutation class regardless of the random draw. Mutation-finite quivers are
  vanishingly rare — 428 of 42.5 M at rank 6 — and are the mathematically
  interesting ones, so they are never left to chance.

Class exploration is an **unlabeled** BFS at `EXPLORE_BOUND = 2`:

- A mutation reaching `|b_ij| >= 3` proves the class mutation-infinite at rank
  ≥ 3 (Derksen–Owen), and the walk records it as `exploration = 'bound'`.
- A walk that drains within the bound proves the class finite (`'complete'`).
- A walk stopped by `--node-cap` is `'truncated'`: finiteness **unknown**, never
  finite. Three-state throughout; soundness over completeness.

## Known results (sanity checks)

| Quiver type | Distinct quivers in the class | Labeled matrices |
|---|---|---|
| A2 | 1 | 2 |
| A3 | 4 | 14 |
| D4 | 6 | 50 |
| A6 | 49 | — |
| D6 | 80 | — |
| E6 | 67 | — |
| E8 | 1,574 | ~6×10⁷ |

The E8 figure is why class discovery is an unlabeled BFS: its labeled orbit is
tens of millions of matrices, its unlabeled one is 1,574 quivers.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # pytest only; the pipeline is stdlib-pure

python -c "
from qmd.core import run_generation
r = run_generation(max_vertices=4, bound=2)
print(f'{len(r.quivers)} quivers in {len(r.classes)} mutation classes')
"

python -m pytest tests/ -q        # 128 tests, incl. the golden-ID guard
```

CI (`.github/workflows/ci.yml`) runs pytest, typecheck and the API smoke tests
against a freshly generated local D1 on every push and PR.

## Contact

Maintained by Blake Jackson — <jackson@icarm.io>

## License

CC-BY-4.0
