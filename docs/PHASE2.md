# Phase 2 design: n ≤ 10, |b_ij| ≤ 10, agents, nicknames

> Status: **implemented** (schema v2, pipeline, API, site, census generation). `docs/SCALING.md` is the original audit; this file records what
> was decided and built.

## 0. What phase 2 has to survive

Concrete numbers that drive every choice below (Cloudflare D1 limits as of
Aug 2026, and the sizes of the finite families):

| Limit / size | Value | Consequence |
|---|---|---|
| D1 max row / string | **2 MB** | one JSON blob per class orbit is impossible (D₁₀ orbit ≈ 136k matrices ≈ 40 MB) |
| D1 max SQL statement | **100 KB** | export must chunk INSERTs by *bytes*, not row count |
| D1 max database | 10 GB (paid) | all A/D/E + affine families to rank 10 fit (≈ 1–3 M labeled rows) |
| D1 max `d1 execute --file` | 5 GB | per-rank files must be split into parts anyway for resumability |
| D1 query duration | 30 s | no query may sort or scan an unbounded set; every list is index-ordered + keyset-paged |
| Worker memory | 128 MB | never materialise an orbit in the Worker; stream everything |
| D₁₀ / E₈ / A₁₀ orbits | 136,136 / 25,080 / 58,786 labeled | class members must be paginated in the API *and* the UI |

## 1. Generation scope — decided: bounded-height census where it is finite, sampling beyond

The maintainer wants a large ML dataset, i.e. a census over (n, h) cells rather
than curated families. The exact cell sizes (Burnside over Sₙ,
`qmd.census.count_quivers`, verified against brute force) are:

| n \ h | 1 | 2 | 3 | 4 | 5 | 10 |
|---|---|---|---|---|---|---|
| 3 | 7 | 25 | 63 | 129 | 231 | 1,561 |
| 4 | 42 | 695 | 5,012 | 22,365 | 74,206 | 3,576,111 |
| 5 | 582 | 82,880 | 2,364,495 | 29,102,787 | 2.2e8 | 1.4e11 |
| 6 | 21,480 | 42,598,925 | 6.6e9 | 2.9e11 | 5.8e12 | 9.5e16 |
| 7 | 2,142,288 | 9.5e10 | 1.1e14 | 2.2e16 | 1.5e18 | 1.2e24 |
| 8 | 5.8e8 | 9.2e14 | 1.1e19 | 1.3e22 | 3.6e24 | 2.6e32 |
| 10 | 8.2e14 | 7.8e24 | 3.0e31 | 2.4e36 | 2.0e40 | 8.7e52 |

So a *census* is a finite job only for cells of roughly ≤ 10⁷ quivers:
(4, h ≤ 10), (5, h ≤ 3), (6, h ≤ 2), (7, 1). Everything else — in particular
every cell with n ≥ 7, h ≥ 2 — is only reachable by **sampling**. Both are
supported by the same pipeline (`scripts/populate.py`):

* `--generator orderly` (default): canonical augmentation (`qmd/census.py`),
  exact, emits every class once, parallel over parents. (5, 2) takes ~25 s on
  8 workers; (6, 2) is ~4×10⁷ classes and is a multi-hour job with a memory
  plan of its own (SCALING §3f) — not run yet.
* `--generator sample --sample N`: N distinct quivers from uniformly random
  labeled matrices of the cell, canonicalised. Uniform over *labeled*
  matrices, not over isomorphism classes (symmetric quivers under-represented)
  — stated in `rank_stats.generator` and on `/api/stats` so ML users know.
* `--node-cap C`: every class BFS stops at C labeled matrices. A class that
  crossed the weight bound before that is still `bound` (proved infinite);
  one that did not is `truncated` (finiteness unknown). This is what makes a
  census of mostly-infinite classes storable.
* `--workers W`: generation, BFS and per-class searches run in a process
  pool. Results are bit-identical to a serial run (the parent replays the
  sequential coverage rule), so published ids never depend on W.

Per-rank provenance (`rank_stats`: bound, node cap, generator, exact
`census_size`, pipeline version, date) tells a reader exactly what fraction of
a cell a rank contains.

## 2. Storage: schema v2 (`drizzle/0001_phase2.sql`)

* **`labelings`** `(mutation_class_id, ord, qmd_id, matrix)` — one row per
  labeled exchange matrix. `ord` is the position in the class's lex-sorted
  orbit (deterministic across Python versions). Indexes: `(qmd_id, ord)` for
  "labelings of this quiver", `(mutation_class_id, qmd_id, ord)` for
  per-class member pages. Replaces `mutation_class_payloads.labeled_quivers`.
* **`frontier_quivers`** `(mutation_class_id, ord, matrix)` — replaces
  `boundary_quivers` JSON for the same reason.
* `mutation_class_payloads` is dropped.
* **`mutation_classes.exploration`** ∈ `complete | bound | truncated`:
  `complete` = BFS drained (finite, exact `class_size`); `bound` = a mutation
  crossed the weight bound (for rank ≥ 3 this proves mutation-infinite); 
  `truncated` = the node cap stopped the BFS — **finiteness unknown**, no
  Dynkin label, no `false` for mutation-acyclicity, `class_size` = explored
  size only. `is_open` stays as the compatibility flag (`exploration != complete`).
* **`quivers.labeling_offset`** — prefix sum of `labeling_count` in `(n, id)`
  order, so the `labelings` scope of a list can be windowed by index instead of
  walking every row.
* **`class_nicknames`** `(mc_id, nickname, slug, note, added_by, added_at)` —
  curated, *never* touched by per-rank imports (see §5).
* **`rank_stats`** gains `bound`, `node_cap`, `generated_at`, `pipeline_version`
  so the site can state exactly how each rank was produced.
* Composite indexes match the actual query shapes: `(n, max_edge, id)`,
  `(n, class_size, id)`, `(n, dynkin_type, id)`, `(n, distinct_quiver_count, id)`,
  `(mutation_class_id, labeling_count DESC, id)`. The never-filterable
  single-column boolean indexes were dropped.

## 3. Pipeline (`qmd/`)

* `_bfs_orbit(seed, bound, node_cap)` records `truncated`; `_merge_orbits`
  derives `exploration`. Gluing treats truncated orbits like open ones
  (partial explorations may share quivers); closed+anything is still a bug.
* `d1_export.render_rank_sql` is now a **streaming writer**: statements are
  cut at 90 KB, files at 64 MB (`qmd-n{k}.001.sql`, `.002.sql`, …). Part 001
  carries the rank's DELETEs, so a rank is idempotent as long as its parts are
  imported in order (`scripts/import-d1.sh`). The manifest lists every part's
  sha256 **and** the sha256 of each lower-rank acyclicity checkpoint it
  consumed, so regenerating rank k invalidates every rank above it.
* Finiteness provenance now states the real threshold
  (`|b_ij| >= bound+1`), and refuses `bound < 2`.
* Seed enumeration is orderly generation (`qmd/census.py`) or sampling; the
  brute-force enumerator remains only as a test oracle.

## 4. API (`src/api/`)

* **Keyset cursors everywhere.** Every list response carries `next_cursor`
  (opaque, `null` at the end); pass it back as `?cursor=`. `offset` still
  works for the page-number UI. Bulk pulls must use cursors.
* `GET /classes/{id}` no longer embeds the orbit. It returns the first page of
  `distinct_quivers` (+ `distinct_quivers_next_cursor`), and `labeled_quivers`
  only when `labeled_size ≤ 200` (otherwise `labeled_quivers_truncated: true`).
  Paged endpoints: `GET /classes/{id}/quivers`, `GET /classes/{id}/labelings`,
  `GET /quivers/{id}/labelings`.
* `scope=labelings` lists are served from the `labelings` table with the
  default `(n, id, ord)` order; other sorts return 400 in that scope (the UI
  disables them) rather than sorting a million rows on the fly.
* Totals come from `rank_stats` whenever the only filter is `rank`.
* `GET /export.ndjson` — resumable bulk pull: one JSON object per line, keyset
  paged, `X-Next-Cursor` header. CSV export is keyset-paged internally too
  (no more OFFSET, no duplicated rows during a re-import).
* `GET /openapi.json` — OpenAPI 3.1 for the whole API; `GET /nicknames`.
* `Access-Control-Allow-Origin: *` on `/api/*` (read-only public dataset) and
  `Cache-Control: public, max-age=300` on lists/details (data changes only on
  ingest); exports are `no-store`.

## 5. Nicknames (curated, survives regeneration)

Source of truth is **`data/nicknames.json` in git**, not the database:

```json
{ "mc_id": "MC.n3.7405511b230b7552", "nickname": "Markov", "slug": "markov",
  "matrix": [[0,2,-2],[-2,0,2],[2,-2,0]], "note": "…", "added_by": "Blake Jackson", "added_at": "2026-08-27" }
```

* `python scripts/nicknames.py --check` validates the file (id format, unique
  slugs, and — the important one — that `quiver_id(matrix)` really belongs to
  `mc_id` in the current dataset). CI runs it.
* `python scripts/nicknames.py --sql dist/nicknames.sql` renders an idempotent
  full-replace of `class_nicknames`; import it after any rank import. Because
  the file carries a member matrix, a nickname can be **re-resolved** after a
  re-keying (`--resolve`) instead of being lost.
* Who may add one: whoever can merge to `main`. That is the "controlled way":
  reviewed, versioned, and independent of the database's lifetime.
* Surfaced as `nickname`/`slug` on class rows and detail, `/api/classes/by-slug/{slug}`,
  `/class.html?name=markov`, and in Browse/Search next to the class id.

## 6. Agents

* `/mcp` — a stateless Model Context Protocol server (Streamable HTTP, no auth)
  with tools `search_quivers`, `get_quiver`, `get_class`, `list_class_members`,
  `get_stats`, `list_nicknames`. Any MCP-capable agent can be pointed at
  `https://quivermutationdb.org/mcp`.
* `/llms.txt` — the dataset, ID scheme, endpoints, cursors, citation, licence,
  in the form agents read first.
* `/api/openapi.json` for tool-calling clients that prefer REST.

## 7. Site

* Class page: members are paged (100 per page, "load more"), grid view is
  capped per page, "All labelings" is served from `/labelings`.
* Quiver page: members table paged the same way.
* Browse: sort headers disabled in labelings scope; nickname shown.
* Home: counts and ranks derived from `/stats`; nothing hard-coded.

## 8. Rollout order for the real data

1. `npm run db:migrate:remote` (schema v2 — additive except the payloads drop).
2. `python scripts/populate.py --export-d1 dist/d1 --max-vertices N --bound B [--node-cap C]`
3. `scripts/import-d1.sh dist/d1 --remote` (parts in order, ranks ascending).
4. `python scripts/nicknames.py --sql dist/nicknames.sql && wrangler d1 execute qmd --remote --file=dist/nicknames.sql`
5. `npm run deploy`.
Regenerate `tests/golden/` only for a deliberate re-keying, with an alias table.
