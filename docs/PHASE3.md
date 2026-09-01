# Phase 3: the census build — cells, cost, sharding, unlabeled exploration

> Status: **built** (schema v3, pipeline, API, site). Generation runs and the
> production load are operated by the maintainer (§6). PHASE2.md remains the
> design of the storage/API foundations; this file records the census plan
> and the three findings that reshaped it.

## 1. What is being built

The maintainer wants a large ML dataset with E8 in it, at ~$10–20/month on
the ICARM Cloudflare account. The agreed sliding scale (all **connected**
quivers only — a disconnected quiver is a disjoint union):

| cell (n, h) | connected quivers | classes explored | labelings |
|---|---|---|---|
| (4, 10) | 3,574,495 | seeds with max\|b\| ≤ 2 (the (4,2) cell); the rest are Derksen–Owen-infinite, no BFS | finite classes |
| (5, 3) | 2,359,306 | same rule ((5,2) subset) | finite classes |
| (6, 2) | 42,514,454 | (c): a cheap capped BFS from every seed to *label* finiteness, class rows only for a 1 M sample | finite classes in the sample |
| (7, 1) | 2,120,098 | all, node cap 100 | finite classes |
| (8, 1) | 1 M sample + curated seeds (E8, D8, A8, E7+A1, E6+A2) | all sampled, node cap 100 | finite classes |

## 2. Three findings that changed the design

1. **Labeled orbits explode.** E6's class has 67 quivers but 42,840 labeled
   matrices; E8 would be ~1,574 × 8! ≈ 6×10⁷. Exploring or storing labeled
   orbits at rank 8 is impossible. Class discovery is therefore an
   **unlabeled BFS** (`qmd/core._bfs_unlabeled`): every mutation result is
   canonicalised, each quiver is visited once. Mutation commutes with
   relabeling, so membership and the MC id (lex-min over members) are
   unchanged — verified against the published n ≤ 4 ids. E8: 1,574 quivers in
   20 s. Labeled orbits are computed only for complete classes with
   `distinct × n! ≤ 200,000` (`LABELED_MAX`), which is where labelings are
   stored (`labelings_stored` in the API, `class_size` NULL otherwise).
2. **Derksen–Owen makes the big cells cheap.** A rank ≥ 3 quiver with an entry
   |b_ij| ≥ 3 is mutation-infinite with no exploration. 99.9 % of (4,10) and
   97 % of (5,3) are labelled this way; only their (n,2) sub-cells need a BFS.
   Consequently the exploration bound is a **constant 2** (`EXPLORE_BOUND`):
   the wall at |b_ij| = 3 *is* the Derksen–Owen witness, and a wall crossing
   always wins over the node cap.
3. **Sparse quivers broke the canonicaliser.** Isolated or twin vertices made
   the lex-min search enumerate thousands of identical branches (6 s for a
   single arrow on 10 vertices). Twin-vertex pruning fixed it (0.5 ms), with
   brute-force agreement on random sparse matrices.

## 3. Storage: schema v3 and sharding

* `drizzle/0003_census_v3.sql` recreates the row tables: compact
  upper-triangular matrix encoding (`qmd/encoding.py`, `src/db/matrix.ts`),
  rowid-based indexes (rows are inserted in id order per rank, so
  `(n, rowid)` is id order and cursors use rowid as the tiebreak — roughly
  half the index bytes), per-quiver `mutation_finite`, `mutation_class_id`
  NULL for unexplored quivers, `class_size` NULL when the labeled orbit is
  not stored, no frontier rows, no cross-table FK on quivers.
* Measured bytes per row: quivers ≈ 400–450 (with the old indexes) → ≈ 230
  now; mutation_classes ≈ 650; labelings ≈ 240.
* **Sharding** (`data/shards.json`, `src/db/shard.ts`): one main database
  plus, for a split rank, `buckets` databases chosen by the first hex digit of
  the id hash. Rank 6 is split in **four** (`qmd-n6-0..3`, created in the
  ICARM account) so each shard's initial load (~10.6 M quivers × 4 row-writes
  ≈ 42 M) fits one month's 50 M included D1 writes; `scripts/trim-shard-indexes.sh`
  drops the (n)-prefixed indexes a single-rank shard does not need. Lists query every shard that can hold matching rows and
  merge by sort key (`src/api/merge.ts`, composite cursors); a class row and
  its labelings live in the shard of the class id; members of a class may
  span shards, so member lists merge too. `rank_stats.shard_counts` lets
  `/random/*` pick a shard proportionally.
* **Cost:** ≈ 3 GB browseable cells + ≈ 10 GB rank 6 → ≈ $12–15/month on the
  Workers Paid plan. D1 bills one row-write per index touched (7 per quiver
  with the full index set, 4 on a trimmed shard). Load schedule against the
  50 M included writes per cycle: ranks 1–5 (≈ 42 M) in one cycle, ranks 7–8
  (≈ 24 M) plus one rank-6 shard (≈ 42 M) the next (~$16), then one shard per
  cycle at $0 (`scripts/import-d1.sh dist/d1 --remote --ranks 6 --shard n6.k`).

## 4. Pipeline

`scripts/populate.py --export-d1 dist/d1 --max-vertices N --bound H [--node-cap C]
[--generator orderly|sample --sample K] [--workers W] [--la-timeout S]`

* Seeds: connected quivers of the cell via orderly generation
  (`qmd/census.py`, exact; counts by Burnside + Euler transform) or a
  uniform sample of labeled matrices (documented bias), plus curated seeds
  (`data/seeds.json`).
* Per seed: Derksen–Owen shortcut, else unlabeled BFS at bound 2 with the
  node cap, in a process pool that reproduces the serial coverage rule
  exactly (published ids never depend on the worker count).
* Per class: Dynkin classification to rank 8 (E6–E8; reference cached in
  `dist/dynkin-reference.json`), Banff/Louise/P′ with a per-open-class
  timeout (`--la-timeout`, default 1 s; 0 = unknown), mutation-acyclicity
  with the component-aware subquiver fallback.
* Export: per-shard part files, byte-bounded statements, manifest with
  checkpoint hashes; `scripts/import-d1.sh` targets the right database per
  part; `scripts/migrate-all.sh` migrates every shard.

## 5. API and site additions

* `/api/lookup?matrix=[[…]]` (and POST): canonicalise a pasted matrix in the
  Worker (`src/canon.ts`, the same lex-min definition), return its id and the
  row if present — "search directs you to the correct unlabeled quiver".
  Also the MCP tool `lookup_quiver` and a box on the Search page.
* New fields: `mutation_finite`, `explored`, `labelings_stored`; `explored`
  filter; class-less quivers render as "Infinite (Derksen–Owen)" /
  "Unexplored"; labeled orbit shows "not stored" where it is not.

## 6. Operating the census (maintainer)

### Running the rank-6 job (learned the hard way)

Three properties of the machine, not the mathematics, cost this stage several
days; `scripts/run-rank6.sh` encodes all three.

* **Never write verdicts into `quivers` row by row.** A scattered
  `UPDATE quivers SET ... WHERE id = ?` against the 7 GB scratch table costs one
  random page read per row. Profiling the first attempt found 90 % of the
  writer's time inside `sqlite3BtreeTableMoveto → readDbPage → unixRead`, at a
  rate that would have needed weeks for a stage whose *compute* is ~2 hours.
  The label stage now appends to `label_verdicts` and folds it into `quivers` in
  one id-ordered pass (`stage_label_apply`), so the B-tree is walked
  sequentially. Every scratch connection also sets `cache_size` to 2 GB — the
  2 MB default guarantees a cache miss on every seek at this scale.
* **Resume from a committed watermark, not a scan.** Scanning for unsettled
  rows re-skips the settled prefix on every batch (quadratic); walking that
  prefix again on each restart cost 13 minutes. `watermarks` records the
  position, `_seed_label_watermark` advances it over settled runs, and
  `mutation_finite` remains the sole ledger of what is settled.
* **`caffeinate -i` does not survive a closed lid.** It blocks only *idle*
  sleep. `run-detached.sh` now uses `caffeinate -s` (system sleep, on AC power);
  a three-day run accumulated under three hours of CPU before this.
* **`ORDER BY id` on a TEXT primary key is not a sequential scan.** SQLite
  satisfies the sort by walking `sqlite_autoindex_quivers_1` and seeking the
  table once per row — the same random-read pathology as writing row by row,
  and `EXPLAIN QUERY PLAN` names it (`SCAN quivers USING INDEX ...` versus a
  plain `SCAN quivers`). Scan unordered and sort the survivors in Python.

### The mutation-finite classes are not a sampling problem

A capped label pass **cannot decide a class larger than its cap**: the BFS
neither drains nor crosses the wall, so every member comes back *unknown*. At
`label_cap = 20` that swallowed all thirteen of rank 6's mutation-finite
classes. Three of them (A6, D6, E6, at 49/80/67 quivers) were rescued only
because the 250k uniform sample happened to land in all three — a ~3 % event —
and the other ten were exported with no `mutation_classes` row at all.

Sampling is the wrong instrument here: 428 of the cell's 42,514,454 quivers are
mutation-finite, so any individual class is a one-in-a-million target. Two
stages fix it, and both are cheap because the survivors are so few:

* `stage_resolve` re-explores whatever is still unknown at `RESOLVE_CAP`
  (100,000). Rank 6 had 492 leftovers and settled all of them in ~30 s. The cap
  has to be large, not merely larger: the slowest infinite one visits 1,089
  quivers before it crosses.
* `stage_finite_classes` explores every mutation-finite class to completion
  (uncapped — finiteness is already proved, so the walk terminates) and stores
  it with its labeled orbit.

Rank 6's complete mutation-finite census is **13 classes over 428 quivers**:
A6 (49), D6 (80), E6 (67); four mutation-acyclic classes at 42/40/36/22 — three
on a 6-cycle (the Ã(p,q) with p+q=6) and one of D̃5 shape; and six
non-mutation-acyclic classes at 48/24/6/5/5/4 (surface and exceptional types).
Together they hold 250,830 labeled matrices.

Never let the finite classes depend on a sample. They are the mathematically
interesting rows in the entire cell.

### Naming them: surfaces, generated not curated

By Fomin–Shapiro–Thurston every mutation-finite quiver of rank ≥ 3 either comes
from a triangulated marked surface or is one of eleven exceptional classes.
`qmd/surfaces.py` makes the surface half constructive: a triangulation is `t`
triangles with a partial matching on their `3t` sides, from which the genus,
boundary components and punctures follow by Euler's formula and the adjacency
quiver from the counter-clockwise arrow rule. Since the flip graph of a surface
is connected, ONE triangulation per surface yields the whole mutation class.

Enumerating surfaces is exact (`n = 6g + 3b + 3p + c - 6`); finding a
triangulation for each is done by seeded random gluing, because the number of
matchings on 3t sides is astronomical by rank 8 while the number of surfaces is
tiny. Coverage is then checked against the enumeration, so a miss is reported
rather than silently dropped. Ranks 3–8 are fully covered in seconds (rank 8
takes ~85 s, almost all of it exploring the classes, and the table is cached).

The result cross-checks against `qmd/dynkin.py` at every rank — the (n+3)-gon is
A_n and the once-punctured n-gon is D_n — and reproduces the curated Markov
nickname as the once-punctured torus. Rank 6's thirteen classes come out as:

| quivers | class |
|--------:|-------|
| 80 | once-punctured 6-gon (**D6**) |
| 67 | **E6** — exceptional, not a surface |
| 49 | 9-gon (**A6**) |
| 48 | once-punctured annulus(1,2) |
| 42 | annulus(1,5) = Ã(1,5) |
| 40 | twice-punctured 3-gon = D̃5 |
| 36 | annulus(2,4) = Ã(2,4) |
| 24 | torus, 1 boundary, 3 marked points |
| 22 | annulus(3,3) = Ã(3,3) |
| 6 | pair of pants(1,1,1) |
| 5 | twice-punctured torus |
| 5 | **X6** — exceptional, not a surface |
| 4 | 4-punctured sphere |

Eleven surfaces plus E6 and X6 is exactly thirteen, which is an independent
confirmation that the census found precisely the right classes. That X6 is
absent from the surface table is required, not incidental: Derksen–Owen proved
it is not block decomposable, and a surface landing on it would mean the module
is wrong. The test suite asserts exactly that.

### Seeding the finite classes instead of hoping for them

A census cell is bounded, and ranks 7 and 8 are taken at `|b_ij| <= 1`. A
mutation-finite class whose every member carries a double arrow could therefore
never be seeded, and rank 8 is sampled besides, where even a class that *is* in
the cell is a one-in-a-million target. `_curated_seeds` closes both holes by
constructing the seeds: every Dynkin and affine-E type of the rank
(`dynkin._seeds_of_rank`, `dynkin.extended_seeds_of_rank`), one quiver per
triangulated surface (`surfaces.seed_quivers`), and the genuinely exceptional
ones from `data/seeds.json`. Exploration runs at `EXPLORE_BOUND = 2`, so each
seed drags its whole class in, double arrows and all — which is why
`quiver_count` exceeds the cell size at these ranks.

**Exceptional classes can be found by extension.** Mutation-finiteness is
hereditary for full subquivers, so every rank-n mutation-finite quiver restricts
to a mutation-finite one on any n-1 of its vertices; conversely an exceptional
class can be hunted by adding a vertex to the rank below. Derksen–Owen note X6
is a subquiver of X7, and of the 35,344 connected one-vertex extensions of X6's
five members exactly **one** mutation-finite class appears —
`MC.n7.9c0c001292e85304`, with 2 quivers and arrow counts {12, 15}, matching
Du–Li–Pan Thm 3.17 for X7 exactly. Its two members are one simple quiver and
one with three double arrows, so the exhaustive (7,1) cell finds it anyway; it
is in `data/seeds.json` regardless, along with X6, so neither depends on luck.

The pipeline is pure standard-library Python, so the job runs on the system's
native arm64 interpreter rather than the x86_64 venv, which is translated by
Rosetta at roughly 0.6x. The golden n<=4 ids are identical on both
interpreters, so there is no re-keying risk; the test suite still runs on the
venv (pytest is not installed for the system Python).

```bash
python scripts/populate.py --count-only --max-vertices 8 --bound 2      # sizes first
# small cells, all classes explored where |b|<=2:
python scripts/populate.py --export-d1 dist/d1 --ranks 4 --bound 10 --node-cap 100 --workers 8
python scripts/populate.py --export-d1 dist/d1 --ranks 5 --bound 3  --node-cap 100 --workers 8
python scripts/populate.py --export-d1 dist/d1 --ranks 7 --bound 1  --node-cap 100 --workers 8
python scripts/populate.py --export-d1 dist/d1 --ranks 8 --bound 1  --node-cap 100 --workers 8 --generator sample --sample 1000000
scripts/run-rank6.sh          # rank 6 (42.5 M quivers): the streaming job, resumable
scripts/release-data.sh dist/d1                                          # migrations → data → nicknames → deploy
```
