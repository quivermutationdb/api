# Planned migration: D1 → PlanetScale Postgres (+ R2 for bulk downloads)

**Status: DECIDED, NOT STARTED. Do not begin until ranks 7 and 8 have finished
generating.** Nothing has been provisioned and no database has been written to.
The current system is still one Worker over D1 exactly as CLAUDE.md describes.

Decided 2026-09-01 by Blake. Full spec with tables and step-by-step:
<https://claude.ai/code/artifact/8b57ac3a-faf8-4a56-9d48-4831e4e19055>

---

## Why we are leaving D1

Four problems with one root. Sharding rank 6 across four databases was never
about size — `data/shards.json` says it plainly: the split exists so each
shard's load fits one month's 50 M included D1 writes. Rank 6 is 8.3 GB and
would fit in a single 10 GB database.

1. **Cross-shard joins are broken.** A class row lives in its own id's shard
   while its member quivers are spread across all four, and `baseQuery` in
   `src/api/quivers.ts` left-joins `mutation_classes` *within one shard
   database*. Measured on shard n6.0: of 598,352 classed quivers, only 196,537
   have their class row in the same shard — **67.2 % return NULL** for
   `dynkin_type`, `class_size`, `exploration` and `is_open`. Filters and sorts
   on those columns silently under-report.
2. **Nicknames can never show for rank 6.** `class_nicknames` is main-only, so
   the shard-local join finds an empty table. Zero rows, always.
3. **The write tax.** ~$124 to load rank 6 (on a fresh cycle), and because the
   documented workflow for a new invariant is "regenerate and re-import", every
   future invariant costs roughly that again.
4. **The index set is frozen at load time.** D1 caps a query at 30 seconds and
   `CREATE INDEX` on 42.5 M rows will not finish inside it, so indexes must
   exist before the data is inserted. `scripts/trim-shard-indexes.sh` already
   drops three quiver indexes permanently with no restore path.
   *(Inferred from the documented 30 s limit, never tested — verify it if it
   ever becomes load-bearing.)*

Postgres fixes all four: one database (joins work), `COPY` is not billed per
row, `CREATE INDEX CONCURRENTLY` has no time limit, and a release becomes an
atomic table swap instead of hours of partially-loaded data.

## Why PlanetScale specifically

Compared at ~18 GB, always-on, with ~4 GB RAM wanted for the hot index pages
(rank 6's indexes measure ~2.2 GB):

| option | RAM | ~monthly | note |
| --- | --- | --- | --- |
| **PlanetScale PS-40** | **4 GB** | **~$29** | 10 GB storage included, then $0.125/GB |
| Neon, scale-to-zero off | 1 GB floor | ~$25 | 4 GB *sustained* costs $76/mo |
| RDS db.t4g.small | 2 GB | ~$25 | burst credits; VPC exposure |
| Hetzner CX32 | 8 GB | ~$12 | self-managed |
| D1 today | — | ~$4.50 | + the four problems above |

PlanetScale wins on RAM per dollar and on storage price (2.8× cheaper than
Neon, with 10 GB included), is always-on by default with no scale-to-zero to
misconfigure, and has a first-class Hyperdrive integration. QMD is
storage-heavy and compute-light, which is exactly that shape.

Neon's one remaining advantage — branching with an automated schema-merge
workflow — matters less here than it first appears: PlanetScale Postgres
branching exists (deploy requests are MySQL/Vitess only), and our release is a
full COPY-and-swap of the tables, so what we need is the ability to rehearse a
whole reload against a copy, which PlanetScale branching does.

## Billing — provision from the Cloudflare side

Since 2026-06-18, PlanetScale databases **created from the Cloudflare
dashboard** are billed to the Cloudflare account at standard PlanetScale
pricing and appear on the Cloudflare invoice. One bill, no card at PlanetScale.

**This only works if the database is created through Cloudflare**
(dashboard → Storage & databases → Postgres & MySQL, which hands off to
PlanetScale for engine/region/size and returns to create the Hyperdrive
config). A database created directly at planetscale.com bills separately, and
there is no documented way to move it onto Cloudflare billing afterwards.
Get this right on the first click.

## Why we are waiting for ranks 7 and 8

Three inputs are unknown until generation finishes, and Blake asked for exact
numbers rather than guesses:

* **Rank 7's true row count.** (7,1) and (8,1) are the first cells whose bound
  (|b_ij| ≤ 1) sits *below* `EXPLORE_BOUND` (2), so exploring a seed's class
  reaches double-arrow quivers outside the cell and every one gets a row. Every
  previous rank had `quiver_count == census_size` exactly (1,550 / 3,574,495 /
  2,359,306 / 42,514,454); these will not. The overshoot drives disk size,
  storage overage and possibly the cluster tier.
* **Total Postgres footprint.** 22-character text primary keys on ~50 M rows
  plus PG's 23-byte tuple header should land at 15–20 GB against SQLite's 11 GB
  — an estimate, not a measurement.
* **Whether PS-40's half vCPU is enough** for the filtered-count path.

---

## WHEN RANKS 7 AND 8 FINISH — do this, in order

1. **Verify the generation.** `python scripts/verify-export.py dist/d1` — all
   ranks must report OK. Read `manifest.json` for the exact `quiver_count`,
   `class_count` and `labeled_quiver_count` of ranks 7 and 8, and note how far
   `quiver_count` exceeds `census_size` (this is expected; see above).
2. **Check the mutation-finite classes landed.** Ranks 7 and 8 were generated
   with constructed seeds (`_curated_seeds`: Dynkin + affine-E +
   `surfaces.seed_quivers` + `data/seeds.json` which now holds X6 and X7). Rank
   7's exceptional classes should be E7, Ẽ6 and X7 (`MC.n7.9c0c001292e85304`);
   rank 8's should be E8, Ẽ7 and **E6^(1,1), which is NOT seeded** — hunt it
   the same way X7 was found (mutation-finiteness is hereditary for full
   subquivers, so extend rank-7 finite quivers by one vertex) and add it to
   `data/seeds.json`.
3. **Measure the real Postgres footprint.** Load rank 6 (and 7) into a local
   Postgres from `dist/d1`, `ANALYZE`, then read `pg_total_relation_size` per
   table and index. This converts every cost figure above from estimate to
   measurement and settles the cluster tier.
4. **Run the pre-flight code review** (checklist below). Output a diff plan; do
   not change code yet.
5. **Only then provision**, following the step list in the spec artifact.

## Pre-flight code review checklist

2,320 LOC in `src/`. A first pass found these — they are settled:

* `rowid` is the keyset tiebreak throughout (`quivers.ts`, `classes.ts`).
  Postgres has none. Fix: an explicit `seq bigint`, assigned at load with
  `row_number() OVER (PARTITION BY n ORDER BY id)`. Cursor *shape* is
  unchanged; a bigint is also 8 bytes against a 22-char text id, so the
  index-size argument in `schema.ts` gets stronger, not weaker.
* The shard-local join breakage (above) — fixed by collapsing to one database.
* `totalsFor` in `quivers.ts` only takes the `rank_stats` fast path when the
  filter is rank-only; any other filter runs a real `count(*)` over the whole
  table. Today that fans out across four shards in parallel — on one PS-40 it
  is serial on half a vCPU across 42 M rows. **Highest-risk regression.**
* No other SQLite-isms exist: no `json_extract`, `GLOB`, `strftime`,
  `INSERT OR`, and `random.ts` already avoids `ORDER BY RANDOM()`. Only
  `CURRENT_TIMESTAMP` in the schema.

Still to audit, in rough order of how likely they are to bite:

* **NULL ordering.** SQLite sorts NULLs first ascending; Postgres sorts them
  **last**. Every sort over a nullable column (`class_size`, `dynkin_type`,
  `mutation_finite`) changes page order unless `NULLS FIRST` is specified.
  Silent — no error, just different pages.
* **Text collation.** Keyset predicates compare ids as text. Postgres' default
  collation is not byte order; use `COLLATE "C"` or pagination will skip rows
  at page boundaries. Also silent.
* Boolean/JSON coercion: every `mode:"boolean"` column and JSON round-trip.
* Integer widths — `labeled_quiver_count` and friends can exceed 2^31; use
  `bigint`.
* `export.ts` streaming under a pooled connection.
* Error surfaces in `errors.ts`; connection exhaustion is a new failure mode.
* CI: `npm run test:api` runs against `wrangler dev` + local D1 and will need a
  Postgres service instead. pytest and typecheck are unaffected.

## Sticking points

* **Load on a small instance.** `COPY` of ~50 M rows plus ~15 index builds on
  0.5 vCPU is painful. Provision **PS-160** for the load, resize down to PS-40
  after. Confirm PlanetScale's resize downtime characteristics first.
* **Cursor format change.** The composite cursor is `{shardKey: key}` JSON in
  an opaque wrapper; one shard changes the encoded value. Emit the new format
  and reject old cursors with a `400` naming the change.
* **Rank-6 responses change shape** — `dynkin_type`, `class_size`,
  `exploration`, `nickname` start returning values where 67 % were NULL. That
  is the bug fix, but it is a visible behaviour change worth a changelog note.
* **Hyperdrive caching** is on by default (60 s, 1 h max) and is *not*
  invalidated by writes, so after a release the cache serves the previous
  census. Pick deliberately: short `max_age`, or rotate the config as a release
  step.
* **Egress.** PlanetScale includes 100 GB/month then $0.06/GB; a full census
  export is ~12 GB. This is exactly why the bulk corpus goes to R2, where
  egress is free. Keep the database for browse only.

## R2 bulk corpus

Gzipped NDJSON per rank plus a manifest with sha256s. ~12 GB at $0.015/GB-month
with the first 10 GB free and **egress free** ≈ **$0.03/month**. Uploading is a
few hundred multipart operations against a 1 M/month free Class A allowance.
Serving it on a subdomain would need a DNS change, which CLAUDE.md lists as
human-approved.

## Not a risk

**Golden ids are safe.** Every `Q.*`/`MC.*` id is computed offline in Python and
shipped as data — no id is generated by the database. `tests/golden/ids-n4.json`
keeps its meaning and citations do not break.

**Rollback is cheap.** Keep D1 populated and untouched until Postgres is
verified; reverting is one Worker deploy.

## The counter-argument, kept honest

Collapsing rank 6 to a **single D1 database** would fix problems 1 and 2 for
zero engineering, since 8.3 GB fits under the 10 GB cap. It would not fix the
write tax, the frozen index set, or non-atomic releases. If those three stop
mattering, the cheap option is still on the table.
