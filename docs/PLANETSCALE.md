# Planned migration: D1 → PlanetScale Postgres (+ R2 for bulk downloads)

**Status as of 2026-09-07: READY TO PROVISION. Nothing has been provisioned
yet.** The two blockers are cleared — ranks 7 and 8 are generated and verified,
and the pre-flight code review is done and its safe-on-D1 fixes are landed. The
live system is still one Worker over D1 exactly as CLAUDE.md describes.

**CUTOVER COMPLETE 2026-09-09.** quivermutationdb.org serves PlanetScale
Postgres. Worker version `c844dc35-c2e2-450e-80c3-7674cf1a7ae7`; the live
`/api/stats` reports 50,828,164 quivers across all eight ranks, against
5,935,362 across five on D1 an hour earlier. Ranks 6, 7 and 8 are public for
the first time. The five D1 databases still exist, unreferenced, as the
rollback -- deleting them is a human-approved action and has NOT been done.
Remaining: Phase 5 (the R2 bulk corpus).

**Superseded 2026-09-09: PROVISIONED AND LOADED.** The PS-160 exists, the
Hyperdrive config exists (`3102c79867ae4311b621240f7f200bbe`, wired into
`wrangler.jsonc`), and all 50,828,164 rows are loaded, indexed and verified.
See §"Phase 2 as it actually ran". What remains is Phase 3 (the Worker port),
Phase 4 (cutover) and Phase 5 (R2) — plus the resize down to PS-40.

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

## Measured Postgres footprint  (2026-09-07, Phase 0 — no longer an estimate)

All 50,828,164 rows loaded into a local PostgreSQL 17, every index built
fresh, `ANALYZE` run. Counts matched the manifest exactly (50,828,164 quivers /
327,961 classes / 457,125 labelings).

| table | heap | indexes | total |
| --- | ---: | ---: | ---: |
| quivers | 9,005 MB | 4,752 MB | 13.0 GB |
| mutation_classes | 153 MB | 37 MB | 190 MB |
| labelings | 54 MB | 43 MB | 97 MB |
| rank_stats + class_nicknames | 16 kB | 48 kB | 80 kB |
| **total** | **9.66 GB** | **5.07 GB** | **14.73 GB** |

Per-row, for future ranks: **190 B/row heap** and **100 B/row of index** on
`quivers`. Tuple width tracks matrix width almost linearly (156 B at rank 4,
166 B at 5, 182 B at 6, 202 B at 7, 213 B at 8 — about 1.35 B per encoded
character, the excess being 8-byte alignment padding).

Largest single index: **`quivers_pkey` at 1,969 MB**, because it is a 21-char
text key over 50.8 M rows. `idx_q_n_seq` is 1,089 MB — an int4 `seq` index
costing roughly half a text-id one, which is the measured version of the plan's
"the index-size argument gets stronger".

### Verdict: PS-40 is viable

* **Storage.** 14.73 GB against 10 GB included = 4.73 GB overage at $0.125/GB
  ≈ **$0.59/month**, so ~$29.60 all-in. Not a tier driver.
* **RAM.** The hot browse path — `quivers_pkey`, `idx_q_n_seq` and the two
  class equivalents — is **3.08 GB**, which fits PS-40's 4 GB. The remaining
  ~2 GB of filter indexes (`idx_q_n_max_edge`, `idx_q_n_finite`,
  `idx_q_representation_type`, `idx_q_mc_labcount`, 336–351 MB each) stay cold
  until someone filters on them. Tight but workable.
* **The plan's estimate was right on the total and wrong on the split.** It
  guessed 15–20 GB; the heap came in *below* the low end (9.66 GB, close to
  SQLite's 11 GB rather than 1.5x it) while the indexes came in at more than
  double the ~2.2 GB assumed. The consequence is that the risk is RAM
  pressure on the filtered paths, not disk — which makes the `totalsFor`
  regression the thing to watch, exactly as the pre-flight predicted.

## Measured query latency  (2026-09-09, Phase 0b) — and the fix

Footprint is not latency, so the filtered paths were then timed directly, with
`max_parallel_workers_per_gather = 0` as the honest stand-in for half a vCPU.
On an M-series Mac with fast NVMe, so these are a **floor**, not a prediction:

| `totalsFor` fallback | plan | before |
| --- | --- | ---: |
| rank 6 + `mutation_finite` | Index Scan | 32 ms |
| rank 7 + `is_acyclic` | Seq Scan | 5.5 s |
| rank 6 + `is_acyclic` | Seq Scan | 6.0 s |
| rank 6 + `max_edge = 2` | Seq Scan | 8.1 s |
| rank 6 + `is_connected = false` | Seq Scan | 6.0 s |

Every bad row is the same thing: a full scan of the 9 GB `quivers` heap to
produce one number. Warm and cold were identical (6.03 s vs 6.00 s) — at 9 GB
there is no warm. PS-40 makes it worse in both directions at once, since the
heap cannot fit in 4 GB and the filter runs on half a vCPU: **tens of seconds,
a browse page that never paints.**

**A bigger tier does not fix this.** PS-160 buys 4x the CPU for ~4x the price
and turns 40 s into perhaps 12 s — still broken, now expensively. The scan is
9 GB whatever you rent. Three cheap changes fix it instead:

1. **`random_page_cost = 1.1`** (`drizzle-pg/0002_indexes.sql`). The 4.0
   default is calibrated for seek-bound spinning disks and makes the planner
   refuse good index scans. Rank 7 + `is_acyclic`: **5.5 s → 0.65 s**, 8.4x,
   from one setting. Rank 6 still (correctly) prefers a scan, being 84 % of
   the table.
2. **Three partial indexes** on the filterable quiver columns that had no
   `(n, col)` index — `is_acyclic`, `NOT is_connected`, and explored
   (`mutation_class_id IS NOT NULL`). Each covers only the *rare* side of a
   lopsided boolean; the capped count below covers the common side. Together
   they cost **69 MB** against 5.07 GB of existing index.
3. **A capped count** — `TOTAL_CAP = 10_000` in `src/api/quivers.ts`. Counting
   happens inside a `LIMIT TOTAL_CAP + 1` subquery, so the engine stops once
   the answer stops being interesting, and the response sets
   `total_is_lower_bound: true`. `?total=exact` still buys the real figure.

Re-measured on the same data, running the exact SQL Drizzle now emits:

| cut | after | speedup |
| --- | ---: | ---: |
| rank 6 + `is_acyclic`, capped | **11.8 ms** | 508x |
| rank 6 + explored, capped | **65.7 ms** | — |
| rank 6 + `is_connected = false`, capped | **0.017 ms** | 350,000x |
| rank 6 + `is_acyclic`, `total=exact` | **413 ms** | 14.5x |
| rank 6 + explored, `total=exact` | **198 ms** | — |

Note the last two: with the partial indexes the *exact* count is also viable
again, so the cap is a safety net rather than the only thing standing between
the browse page and a timeout. **PS-40 is the steady tier.**

## Phase 2 as it actually ran  (2026-09-09)

Loaded into PlanetScale Postgres **18.6** (not 17 — the local rehearsal was on
17.11; nothing we use is version-sensitive). Role `pscale_api_*`: not superuser,
not the database owner, but `CREATEDB` + `CREATE` on `public`, which is
everything the load needs.

| phase | wall clock |
| --- | ---: |
| ranks 1–5, 7, 8 (1.0 GB) | 15 min |
| rank 6 upload (5.1 GB, 10 chunks) | 46 min |
| rank 6 staging → table + seq (server-side) | 12 min |
| all indexes, `CONCURRENTLY` | 25 min |

**Verified:** counts exact at all 8 ranks (50,828,164 / 327,961 / 457,125 / 21);
`seq` unique, gapless and reproducing id order at every rank; `rank_stats`
reconciled against rows actually loaded; `COLLATE "C"` intact on all 7 id
columns; zero orphan labelings; no invalid indexes; rank-8 finite classes = 19
with E6^(1,1) at 49 quivers; rank-6 finite = 428 in 13 classes; Markov still not
mutation-acyclic. Script: `scratchpad/verify.sql` — worth keeping with the repo.

**Footprint held to within 3 % of the Phase 0 projection:** 9,213 MB heap /
4,902 MB indexes / 14 GB total, against 9.66 / 5.07 / 14.73 GB predicted.
`quivers_pkey` landed at 1,969 MB, matching the local figure exactly.

### Things only the real load could teach

* **`random_page_cost` is already 1.1 on PlanetScale.** The setting §"Measured
  query latency" argues for is their default. `0003`'s `ALTER DATABASE` takes
  its no-op branch (the role is not the owner) — which is what that guard was
  written for.
* **The upload is per-stream limited, not bandwidth limited.** One `\copy`
  sustained 1.1 MB/s; three concurrent sustained 3.62 MB/s aggregate, near
  linear. **Caveat, honestly:** the driver's `wait -n` is unsupported in this
  zsh and fell back to `wait`, so the production run degenerated to roughly
  serial — 46 min rather than the ~20 the measurement implied. Parallelise a
  future load, but verify the harness actually sustains it.
* **Chunk the big rank.** 5.1 GB in one `\copy` means a dropped connection at
  minute 40 costs everything. Ten chunks into one `UNLOGGED` staging table with
  a per-chunk marker makes it resumable, and costs nothing: `seq` is assigned
  afterwards by `row_number() OVER (PARTITION BY n ORDER BY id)`, so chunk
  arrival order is irrelevant by construction.
* **Rank 6's `rank_stats` row is not in the shard databases.** The stats parts
  target the main database, so a shard-only load silently leaves `/stats`
  missing the rank holding 84 % of the census. Load
  `dist/d1/qmd-n6.stats.001.sql` and `qmd-n6.patch-stats.001.sql` explicitly;
  the patch recomputes `class_count` from rows actually present, which in one
  database gives the true 242,981 rather than the hardcoded 242,971.
* **Do not run `pg_n6b/load.sql` after chunking** — it would re-COPY all 5.1 GB.

### Open: 1,749 dangling rank-8 class references

0.68 % of rank-8 quivers named a `mutation_classes` row that was never written
(1,740 distinct ids, all mutation-infinite, all `max_edge = 1`). **Pre-existing
in the generated data** — same 1,749 in the source SQLite and in `dist/d1`.
Patched at the database with `drizzle-pg/0003_rank8_dangling_class_refs.sql`
(NULL the reference; `mutation_finite` deliberately untouched — see the file).
**The exporter-side cause is unresolved and still open**: rank 8 was generated
by an earlier state of `qmd/d1_export.py`, so today's source does not explain
it. Regenerating rank 8 should make `0003` a no-op; that is the test.

### Measured on the real PS-160

| query | PS-160 | local (Phase 0b) |
| --- | ---: | ---: |
| point lookup by id | **2.3 ms** | — |
| browse page 1, rank 6 keyset | **2.3 ms** | — |
| capped count, rank 6 + `is_acyclic` | **20.6 ms** | 11.8 ms |
| capped count, rank 6 + `is_connected = false` | **0.084 ms** | 0.017 ms |
| exact count, rank 7 + `is_acyclic` | **23.9 ms** | 651 ms |
| exact count, rank 6 + `is_acyclic` | 1,334 ms | 413 ms |

Rank 7 beat the local number 27x because that measurement predates the partial
indexes. **These are PS-160 numbers**: the exact rank-6 count used three
parallel workers that PS-40's half vCPU will not have, so expect ~4 s there —
exactly the case `TOTAL_CAP` exists for, and not a path the API takes by
default.

## Load-path defects found by doing it  (all fixed)

Six, none of which are visible on paper. Two would have produced a database
that loads cleanly and serves wrong results, and the sixth could only be found
by pointing the load at something that is not localhost.

0. **`COPY` was server-side, so the load could never have reached PlanetScale.**
   `pg-export.py` emitted SQL `COPY t (...) FROM '/abs/path'`, which makes the
   *server* open that path, and under `--gzip` it emitted
   `COPY ... FROM PROGRAM 'gzip -dc ...'`, which makes the server run a shell
   command. Both need superuser and both assume the data is on the database
   host. That is true of a local Postgres — the server is your laptop, which is
   exactly why the Phase 0 rehearsal passed — and false of every managed
   provider. Replaced with psql's `\copy` client meta-command, which reads the
   file here and streams it as `COPY ... FROM STDIN` with no special grant.
   Verified 2026-09-09 by loading the 692-row dev dataset over TCP into a fresh
   database: 0001 → `\copy` → 0002 → verify, `seq` gapless per rank.

1. **`UPDATE ... SET seq = row_number()` bloats every table ~2x.** MVCC writes
   a new tuple version per update, so it leaves exactly one dead tuple per row:
   `quivers` went 1,434 MB → 2,835 MB (1.98x) on 8.3 M rows, needing
   `VACUUM FULL` and 2x transient disk. It was also the bulk of an **82-minute**
   load. Replaced with `UNLOGGED` staging + `INSERT ... SELECT row_number()`:
   the 42.5 M-row rank 6 then loaded in **8 min 50 s**, i.e. **47x faster per
   row**, with zero dead tuples.
2. **Per-shard export breaks `seq` silently.** `row_number() OVER (PARTITION BY
   n ORDER BY id)` restarts at 1 in each shard, so four rank-6 rows would share
   `seq = 1` and the keyset tiebreak in `src/api/cursor.ts` stops being unique.
   The database loads cleanly and row counts check out; pagination then skips
   and repeats rows. `scripts/pg-export.py` now requires every shard of a rank
   in one invocation. Verified after loading: `seq` is unique, gapless 1..count,
   and `(n, seq)` reproduces id order at every rank including rank 6 across all
   four merged shards.
3. **`rank_stats` collides on its primary key** — it is keyed by `n` and every
   shard carries its own row for the rank, so the second shard's COPY is a
   duplicate-key error. Deduped in the exporter.
4. **The shard SQL parts cannot be concatenated.** Each part 001 opens with
   `DELETE ... WHERE n = 6`, so loading four shards in sequence leaves only the
   last — found by getting 10,630,908 rows where the manifest says 42,514,454.
   Load each shard into its own SQLite, then merge at the COPY step. The four
   then sum exactly and their ids are disjoint, which is the cross-shard
   property D1 was breaking.
5. **A primary key is an index and bloats like one.** Leaving only the PKs in
   place during the rank-6 insert took `quivers_pkey` from 1,969 MB to
   3,287 MB (1.67x, from page splits); `REINDEX` reclaimed **1.40 GB** across
   the three tables. `drizzle-pg/0001_init.sql` therefore declares no primary
   keys at all on `quivers`/`mutation_classes`/`labelings`, and
   `0002_indexes.sql` adds them as `CREATE UNIQUE INDEX CONCURRENTLY` +
   `ALTER TABLE ... ADD PRIMARY KEY USING INDEX`, with the labelings foreign
   key added `NOT VALID` then validated separately. The three tiny tables keep
   their keys inline.

**Also stale:** `dist/d1/nicknames.sql` holds 1 entry against the 21 now in
`data/nicknames.json`. Regenerate with `python scripts/nicknames.py --sql`
before any load, or the curated names — E6^(1,1) included — silently do not
ship.

## Still open

1. **Provision**, following §"Provisioning runbook" below. Phase 0 is done:
   `drizzle-pg/*.sql` and `scripts/pg-export.py` are written and exercised
   against a full local load.

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
  throw. It was flagged here as the highest-risk regression, with "cap it with
  a `LIMIT` subquery" as the escalation. Measurement (§"Measured query
  latency") confirmed the risk was real — 6 s single-threaded on a machine far
  faster than a PS-40 — so **the escalation was taken up front rather than held
  in reserve**: capped by default at `TOTAL_CAP = 10_000`, `?total=exact` to
  opt out, plus three partial indexes and `random_page_cost = 1.1`. Worst
  measured browse cut is now 66 ms. **Closed.**
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
  is serial on half a vCPU across 42 M rows. ~~**Highest-risk regression.**~~
  Measured and fixed, 2026-09-09; see §"Measured query latency".
* No other SQLite-isms exist: no `json_extract`, `GLOB`, `strftime`,
  `INSERT OR`, and `random.ts` already avoids `ORDER BY RANDOM()`. Only
  `CURRENT_TIMESTAMP` in the schema.

## Provisioning runbook

Ordered so that everything reversible happens before anything that is not, and
so the one irreversible billing decision is the very first click.

### Phase 0 — DONE  (2026-09-07)

Measured, written and exercised against a full local load:
`drizzle-pg/0001_init.sql`, `drizzle-pg/0002_indexes.sql`,
`scripts/pg-export.py`. Footprint and the five load-path defects are above.
Reproduce with:

```bash
brew install postgresql@17            # local only; brew uninstall reverses it
# one SQLite per shard, since the parts each DELETE the rank (defect 4)
python scripts/pg-export.py OUT tmp/qmd_n6_s0.sqlite ... tmp/qmd_n6_s3.sqlite
psql -d qmd -f drizzle-pg/0001_init.sql
psql -d qmd -f OUT/load.sql           # COPY -> staging -> INSERT with row_number()
psql -d qmd -f drizzle-pg/0002_indexes.sql
psql -d qmd -c ANALYZE
```

1. **Decide three things** (they are inputs to the clicks below, and changing
   them later is work):
   * **Region.** Match the Worker's traffic. Cross-region adds latency to every
     query that misses the Hyperdrive cache.
   * **Hyperdrive `max_age`.** Caching is ON by default (60 s, 1 h max) and is
     **not** invalidated by writes, so after a release the cache serves the
     previous census. Either keep `max_age` short, or rotate the config as a
     release step. Pick deliberately.
   * **Steady tier**, from the measurement.
2. **Port `src/db/schema.ts` to `drizzle-orm/pg-core`** so the Worker's types
   match `drizzle-pg/0001_init.sql`. The raw DDL exists; the Drizzle
   declaration does not, and it is what the query builder needs.
3. **Regenerate `dist/nicknames.sql`** — the copy on disk is stale (see above).
   *(Done 2026-09-09: 1 entry → 21.)*
4. **Landed 2026-09-09, from the latency measurement:** `random_page_cost` and
   the three partial indexes in `drizzle-pg/0002_indexes.sql`; the capped count
   (`TOTAL_CAP`, `?total=exact`, `total_is_lower_bound`) across
   `src/api/quivers.ts`, `errors.ts`, `openapi.ts`, `mcp.ts`, `browse.html`,
   `search.html`, `download.js`; smoke coverage in `scripts/api-smoke.mjs`.
   The suite passes with `TOTAL_CAP` forced to 3, which is how the lower-bound
   branch is exercised on a 692-row dev dataset — do that again after touching
   `totalsFor`.

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

### Phase 2 — load  (into a bare heap; every index afterwards)

**Phase 2 needs the direct PlanetScale connection string, not Hyperdrive.**
Hyperdrive is a Worker binding: `env.HYPERDRIVE.connectionString` exists only
inside a running Worker and points at Cloudflare's pooler. `psql` cannot use
it. The Hyperdrive id (`3102c79867ae4311b621240f7f200bbe`, created 2026-09-09)
is wired into `wrangler.jsonc` for Phase 3 and is not needed before then.

**Budget for the upload.** The rank-6 TSV alone is 5.1 GB and the whole census
is ~5.5 GB, all of which crosses the wire from the machine running `psql`.
`--gzip` shrinks the files on disk but not on the wire — `\copy` decompresses
client-side and the Postgres protocol does not compress — so plan on a wired
connection and a run measured in tens of minutes. Load rank by rank: each rank
is a separate export directory, so a dropped connection costs one rank, not
the census.

9. `psql -f drizzle-pg/0001_init.sql`. This creates **no indexes at all, not
    even primary keys** — see defect 5. Do not "helpfully" add them back.
10. For each rank, build one SQLite per shard from `dist/d1` (the parts each
    `DELETE` the rank, so they cannot share a database — defect 4), then
    `python scripts/pg-export.py OUT <every shard of that rank>`. Passing
    shards separately silently breaks `seq` — defect 2.
11. `psql -f OUT/load.sql`. This COPYs into `UNLOGGED` staging and then
    `INSERT ... SELECT row_number() OVER (PARTITION BY n ORDER BY id)`. Do
    **not** substitute an `UPDATE` — defect 1, ~2x bloat and ~47x slower.
12. `psql -f drizzle-pg/0002_indexes.sql` — primary keys, the labelings foreign
    key, and all secondary indexes, all `CONCURRENTLY`. No 30-second limit
    here; that is one of the four reasons for leaving D1.
13. `ANALYZE`, then check against the Phase 0 measurement: expect **9.66 GB
    heap / 5.07 GB indexes / 14.73 GB total** and `n_dead_tup = 0` everywhere.
    Dead tuples mean something did an `UPDATE` it should not have.
14. Verify `seq` before trusting pagination:

    ```sql
    SELECT n, count(*), count(DISTINCT seq), min(seq), max(seq) FROM quivers GROUP BY n;
    ```

    Every rank must satisfy `count = count(distinct) = max` and `min = 1`.

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

* **Load on a small instance.** `COPY` of ~50 M rows plus ~18 index builds on
  0.5 vCPU is painful. Provision **PS-160** for the load, resize down to PS-40
  after — the steady tier is confirmed by §"Measured query latency", not just
  by the footprint. Blake confirmed 2026-09-09 that PlanetScale allows a
  downgrade at any time, so this is reversible in a click if a filtered path
  misbehaves under real traffic.
* **Cursor format change.** The composite cursor is `{shardKey: key}` JSON in
  an opaque wrapper; one shard changes the encoded value. Emit the new format
  and reject old cursors with a `400` naming the change.
* **Filtered list totals become lower bounds.** `total`, `distinct_total` and
  `labeled_total` stop being exact whenever the filter is not rank-only and the
  cut exceeds `TOTAL_CAP`; the new `total_is_lower_bound: true` says so, and
  `?total=exact` restores the old behaviour at the old cost. Additive field, so
  nothing breaks that ignores it — but a client computing a page count from
  `total` will under-count, which is why `browse.html` switches to a prev/next
  pager (driven by whether the last page came back full) when the flag is set.
  Agents are told to page with `next_cursor` in the MCP tool description and
  the OpenAPI note. Worth a changelog line.
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
