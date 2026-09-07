# Planned migration: D1 → PlanetScale Postgres (+ R2 for bulk downloads)

**Status as of 2026-09-07: READY TO PROVISION. Nothing has been provisioned
yet.** The two blockers are cleared — ranks 7 and 8 are generated and verified,
and the pre-flight code review is done and its safe-on-D1 fixes are landed. The
live system is still one Worker over D1 exactly as CLAUDE.md describes.

Remaining before the first dashboard click: **measure the real Postgres
footprint locally** (§"Still open", item 1). Everything else is a decision or a
click, both listed in §"Provisioning runbook".

Decided 2026-09-01 by Blake. This file is authoritative; the original spec
artifact <https://claude.ai/code/artifact/8b57ac3a-faf8-4a56-9d48-4831e4e19055>
predates the measurements below and should not be followed where the two
disagree.

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

## What ranks 7 and 8 actually measured  (2026-09-07)

The numbers we were waiting for. `python scripts/verify-export.py dist/d1`
reports OK on all eight ranks.

| rank | stored rows | classes | labelings | census_size |
| ---: | ---: | ---: | ---: | ---: |
| 1–3 | 1,561 | 19 | 49 | = stored |
| 4 | 3,574,495 | 149 | 392 | = stored |
| 5 | 2,359,306 | 17,567 | 8,874 | = stored |
| 6 | 42,514,454 | 242,981 | 250,830 | = stored |
| 7 | 2,120,315 | 35,241 | 196,980 | 2,120,098 |
| 8 | 258,033 | 32,004 | 0 | 572,849,763 |
| **total** | **50,828,164** | **327,961** | **457,125** | |

**50.8 M rows is the number to size against.** Three traps in that table:

* **The overshoot is tiny, not large.** Rank 7 stores 2,120,315 against a
  census of 2,120,098 — only **217** rows outside the cell, being members of
  complete mutation-finite classes. The worry above (that dropping the cell
  bound below `EXPLORE_BOUND` would balloon the row count) did not materialise,
  because `build_rank_rows` filters explored quivers back to the cell and keeps
  only complete-class members.
* **Rank 8's `census_size` is not a row count.** That cell has 572,849,763
  connected quivers; we store a 250,000 sample plus 8,033 out-of-cell finite
  members. Sizing against 573 M would over-provision by 2000×.
* **Rank 8 has zero labelings**, by the `distinct × n! ≤ 200k` rule: at n=8,
  n! = 40,320, so only classes of ≤ 4 distinct quivers would qualify and none
  are. Not a bug — but rank-8 class pages have no labeled matrices to inline,
  which is a visible product difference from ranks 6–7.

The largest integer anywhere is **572,849,763** (rank 8's `census_size`), which
fits int4 — see the `seq integer` note in the checklist.

---

## Done  (do not redo)

1. ~~**Verify the generation.**~~ Done 2026-09-07: `verify-export: OK` on all
   eight ranks, zero disconnected quivers, zero flag mismatches, every count
   matching the manifest. Numbers in the table above.
2. ~~**Check the mutation-finite classes landed.**~~ Done, and it found a real
   gap. The finite-class counts match the classification exactly at every rank:
   rank 6 = 11 surfaces + E6 + X6 (13), rank 7 = 12 surfaces + E7 + Ẽ6 + X7
   (15), rank 8 = 16 surfaces + E8 + Ẽ7 + **E6^(1,1)** (19).

   Rank 8 first shipped **18** — E6^(1,1) had no constructor anywhere
   (`dynkin._EXTENDED` covers only affine types, `surfaces.py` only surface
   types) and a 250k-of-573M sample was never going to find a 49-quiver class.
   Recovered by exhaustive one-vertex hereditary extension of all 1,892 rank-7
   finite quivers (commit `b33f67c`); it is now in `data/seeds.json` as the
   lex-min member, `MC.n8.e9303af7e4328df4`, 49 quivers, nicknamed in
   `data/nicknames.json`. Rank 8 was regenerated and now reports 19.

   **Only 2 of rank 8's 19 classes were found by the cell sample.** The
   constructed seeds carry essentially the whole finite census at that rank.
4. ~~**Run the pre-flight code review.**~~ Done 2026-09-07; results folded into
   the checklist below. The three fixes that were safe on D1 are landed and
   CI-green (commit `47bdee9`), so they are exercised before the port rather
   than during it.

## Still open

1. **Measure the real Postgres footprint.** The one genuine unknown left. Load
   ranks 6–8 into a local Postgres from `dist/d1`, `ANALYZE`, then read
   `pg_total_relation_size` per table and index. 22-char text primary keys on
   50.8 M rows plus PG's 23-byte tuple header should land at 15–20 GB against
   SQLite's 11 GB — still an estimate. This settles PS-40 vs a larger steady
   tier and the storage-overage figure.
2. **Provision**, following §"Provisioning runbook" below.

## Pre-flight code review checklist  (reviewed 2026-09-07)

2,320 LOC in `src/`. **Landed on D1 already** (commit `47bdee9`, CI green,
59/59 API assertions) — do not redo:

* **NULL ordering — fixed in ONE place, not three.** SQLite sorts NULLs first
  ascending and last descending; Postgres does the exact opposite. Placement is
  now stated explicitly in `orderBy()` (`asc nulls first` / `desc nulls last`).
  `afterKey()`/`strictlyAfter()` and `compareKeys()` **already** encoded
  SQLite's rule correctly, so pinning the emitter keeps all three in agreement
  on either engine; rewriting the predicate logic would have been far riskier
  for the same result. SQLite has accepted `NULLS FIRST/LAST` since 3.30 (2019)
  and the two cursor tests that exercise it pass unchanged.
* **`count(*)`/`sum()` return STRINGS on Postgres — was not on this list.**
  Both are `bigint`, and both Workers drivers hand int8 back as a string (or
  `BigInt`) to protect precision. The `sql<number>` annotations are
  compile-time only, so `a + count` would have *concatenated*: totals arriving
  as `"042514454"` with no error and no type failure. Coerced with `Number()`
  at all four sites (`quivers.ts` ×3, `classes.ts` ×2) rather than a `::int`
  cast, because `::` is not SQLite syntax and would break D1 today.
* **`totalsFor` ran BEFORE the page query**, putting a `count(*)` over two left
  joins on the critical path of every list response. Now started alongside the
  page read, with sort validation hoisted above both so no promise dangles on
  throw. Still the **highest-risk regression** on one PS-40: parallel fan-out
  across four shards becomes serial on half a vCPU over 42 M rows. If it bites,
  escalate: make the exact count opt-out, or cap it with a `LIMIT` subquery
  ("1000+").
* **Typed `Unavailable` → 503 with `Retry-After`** added to `errors.ts` and
  wired into `onError`. Nothing raises it on D1, which has no pool; it exists
  so pooled-Postgres exhaustion does not reach clients (or an agent following
  `/llms.txt`) as an untyped 500 that reads as "give up".

**Deferred INTO the port** — these would break the running D1 system, so they
belong in the migration itself, not ahead of it:

* **`rowid` → an explicit `seq`.** Postgres has no rowid. Assign at load with
  `row_number() OVER (PARTITION BY n ORDER BY id)`. Touches `classes.ts`,
  `export.ts`, `quivers.ts` and the comments in `schema.ts`/`cursor.ts`. Cursor
  *shape* is unchanged.

  **Use `integer`, not `bigint`** (correcting the original plan). Max rows in
  any rank is 42.5 M and the largest counter anywhere is 573 M — both well
  inside int4. That halves the index width *and* keeps the
  bigint-arrives-as-a-string trap out of cursor keys, where `compareKeys`
  compares numerically. A `bigint seq` would silently turn keyset comparison
  into string comparison.
* **Delete the shard machinery** once the databases collapse: `src/db/shard.ts`
  (98 LOC), `src/api/merge.ts` (104), `compareKeys`, `data/shards.json`,
  `scripts/trim-shard-indexes.sh`. Keep `shardFor(n)` as the routing seam
  CLAUDE.md requires — it becomes a one-liner.

Settled by inspection, no action needed:

* The shard-local join breakage (above) — fixed by collapsing to one database.
* **Text collation: the fix is right, the stated mechanism was wrong.**
  `ORDER BY` and the keyset predicate both compile without an explicit
  collation, so both use the database default and are *mutually consistent* —
  pagination does **not** skip rows; only the page *order* changes versus
  SQLite's byte order. The genuine skip risk was `compareKeys` merging shard
  pages in JavaScript (UTF-16 code-unit order) against a linguistic DB
  collation — and that path dies with sharding. Declare `COLLATE "C"` on the id
  columns anyway: it restores byte order (matching SQLite, the documented id
  ordering, and JS), and keeps the index usable by the predicate.
* **JSON is safe on the read side.** `provenance` and `symmetry_group` are only
  ever accessed as parsed objects (`.order`, `.name`), never compared as text,
  so `jsonb` works. One cosmetic consequence: `jsonb` does not preserve key
  order, so response *bytes* change where `provenance` is passed through whole
  (`classes.ts:268`). Semantics unchanged.
* **Integer widths.** Largest live value is 572,849,763 — fits int4, but within
  4× of the ceiling, so a rank-9 cell would exceed it. `bigint` for
  `rank_stats` counters is cheap forward-safety; `seq` stays `integer` (above).
* No other SQLite-isms exist: no `json_extract`, `GLOB`, `strftime`,
  `INSERT OR`, and `random.ts` already avoids `ORDER BY RANDOM()`. Only
  `CURRENT_TIMESTAMP` in the schema.
* CI: `npm run test:api` runs against `wrangler dev` + local D1 and will need a
  Postgres service instead. pytest and typecheck are unaffected. Note
  `test:api` does **not** start the dev server — it expects one on :8787.

### Superseded first-pass notes

The first pass listed NULL ordering, collation, boolean/JSON coercion, integer
widths, `export.ts` streaming and error surfaces as "still to audit". All are
now resolved above except `export.ts` streaming under a pooled connection,
which cannot be tested until a pool exists — check it during the port.

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

## Provisioning runbook

Ordered so that everything reversible happens before anything that is not, and
so the one irreversible billing decision is the very first click.

### Phase 0 — before the dashboard  (no Cloudflare involved)

1. **Measure the footprint** (§"Still open" item 1). Do not pick a steady-state
   tier from the estimate.
2. **Decide three things** (they are inputs to the clicks below, and changing
   them later is work):
   * **Region.** Match the Worker's traffic. Cross-region adds latency to every
     query that misses the Hyperdrive cache.
   * **Hyperdrive `max_age`.** Caching is ON by default (60 s, 1 h max) and is
     **not** invalidated by writes, so after a release the cache serves the
     previous census. Either keep `max_age` short, or rotate the config as a
     release step. Pick deliberately.
   * **Steady tier**, from the measurement.
3. **Write the pg schema** (`drizzle-orm/pg-core`), with the decisions already
   settled in the checklist: `seq integer`, `COLLATE "C"` on id columns, `jsonb`
   for `provenance`/`symmetry_group`, real `boolean` (preserving three-state
   NULL), `bigint` for `rank_stats` counters.
4. **Write the COPY exporter.** This does not exist yet. `dist/d1/*.sql` is
   `INSERT` statements for D1; Postgres wants `COPY ... FROM STDIN` with
   TSV/CSV per table. Generate it from the same `d1_export` row builders so
   there is one source of truth, not a SQL-to-CSV parser.

### Phase 1 — create the database THROUGH Cloudflare  (billing: get this right first)

5. Cloudflare dashboard → **Storage & databases → Postgres & MySQL** → create,
   choosing **Postgres (PlanetScale)**. This hands off to PlanetScale for
   engine/region/size and returns to Cloudflare to create the Hyperdrive config.
6. Size **PS-160 for the load**, not the steady tier — `COPY` of 50.8 M rows
   plus ~15 index builds on PS-40's half vCPU is the difference between hours
   and a weekend.
7. **Verify the billing path immediately**: the database must appear under the
   Cloudflare account with no separate card at PlanetScale. A database created
   at planetscale.com bills separately and there is **no documented way to move
   it onto Cloudflare billing afterwards** — if this is wrong, delete and redo
   before loading any data.
8. Note the Hyperdrive config id and the connection string.

### Phase 2 — load  (indexes last)

9. Apply the schema. **Create no indexes yet** beyond primary keys.
10. `COPY` each table: `quivers`, `mutation_classes`, `labelings`,
    `class_nicknames`, `rank_stats`. Ranks ascending.
11. Populate `seq`: `row_number() OVER (PARTITION BY n ORDER BY id)`.
12. `CREATE INDEX CONCURRENTLY` for each index in `schema.ts`. No 30-second
    limit here — this is one of the four reasons we are leaving D1.
13. `ANALYZE`, then re-read `pg_total_relation_size` and compare against the
    Phase 0 measurement.

### Phase 3 — the Worker  (still not serving Postgres)

14. Add the Hyperdrive binding to `wrangler.jsonc`; `npm run cf-typegen`.
15. Swap the Drizzle driver to pg over Hyperdrive.
16. Apply the two deferred pre-flight items: `rowid` → `seq integer`, and
    delete the shard machinery (keeping `shardFor(n)` as the seam).
17. Check `export.ts` streaming under the pool — the one audit item that could
    not be tested without a pool.
18. Point CI's `test:api` at a Postgres service. Remember it expects a dev
    server already running on :8787.
19. Deploy to a **preview** first and run `npm run test:api` against it.

### Phase 4 — cutover

20. Confirm PlanetScale's **resize downtime characteristics**, then resize
    PS-160 → the steady tier. Do this before production traffic, not after.
21. Deploy to production. Watch the filtered-count path (`totalsFor`) first —
    it is the predicted regression.
22. Leave D1 untouched as the rollback (ranks 1–5; see above).

### Phase 5 — R2 bulk corpus  (independent; can happen any time)

23. Create the bucket from the Cloudflare dashboard (same account, so billing is
    already right).
24. Generate gzipped NDJSON per rank plus a manifest with sha256s (~12 GB).
25. Upload multipart — a few hundred Class A operations against a 1 M/month
    free allowance.
26. **Serve it through a Worker route, not a new subdomain.** A subdomain needs
    a DNS change on quivermutationdb.org, which CLAUDE.md lists as
    human-approved and never automatic.

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

**Rollback is cheap, but read what it rolls back TO.** Reverting is one Worker
deploy — however production D1 only ever received **ranks 1–5**. Ranks 6, 7 and
8 were generated but deliberately never imported (that decision, 2026-09-06,
is what saved the ~$124–166 of D1 writes). So rollback restores a working site
covering ranks 1–5, not the full census. Keep D1 untouched regardless; it is
still the fastest way back to a serving site.

A consequence worth stating plainly: **PlanetScale is not a migration of live
data, it is the first publication of ranks 6–8.** There is no "before" to
compare rank-6 responses against in production.

## The counter-argument, kept honest

Collapsing rank 6 to a **single D1 database** would fix problems 1 and 2 for
zero engineering, since 8.3 GB fits under the 10 GB cap. It would not fix the
write tax, the frozen index set, or non-atomic releases. If those three stop
mattering, the cheap option is still on the table.
