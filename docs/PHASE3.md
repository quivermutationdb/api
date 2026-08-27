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
  the id hash. Rank 6 is split in two (`qmd-n6-0`, `qmd-n6-1`, created in the
  ICARM account). Lists query every shard that can hold matching rows and
  merge by sort key (`src/api/merge.ts`, composite cursors); a class row and
  its labelings live in the shard of the class id; members of a class may
  span shards, so member lists merge too. `rank_stats.shard_counts` lets
  `/random/*` pick a shard proportionally.
* **Cost:** ≈ 3 GB browseable cells + ≈ 10 GB rank 6 → ≈ $12–15/month on the
  Workers Paid plan. One-time row-writes for rank 6 (~130 M with 3 indexes)
  are ~$80–120 if loaded in one month, or free if staged over three months
  (`scripts/import-d1.sh … --shard n6.0`, then `n6.1` the next month).

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

```bash
python scripts/populate.py --count-only --max-vertices 8 --bound 2      # sizes first
# small cells, all classes explored where |b|<=2:
python scripts/populate.py --export-d1 dist/d1 --ranks 4 --bound 10 --node-cap 100 --workers 8
python scripts/populate.py --export-d1 dist/d1 --ranks 5 --bound 3  --node-cap 100 --workers 8
python scripts/populate.py --export-d1 dist/d1 --ranks 7 --bound 1  --node-cap 100 --workers 8
python scripts/populate.py --export-d1 dist/d1 --ranks 8 --bound 1  --node-cap 100 --workers 8 --generator sample --sample 1000000
# rank 6 (42.5 M quivers) is the streaming job: see qmd/bigcell.py
scripts/release-data.sh dist/d1                                          # migrations → data → nicknames → deploy
```
