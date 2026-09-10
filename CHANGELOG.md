# Changelog

Changes to the **published data and the public API**. Internal refactors are not
listed; the git history has those.

This file exists because QMD is meant to be cited. A paper that says "the
mutation class of `MC.n6.…` has 42,840 labeled matrices, per QMD" is making a
claim about a specific state of this database, and a reader needs to be able to
tell whether that state still holds. Two rules follow:

- **Identifiers are frozen.** `Q.*` and `MC.*` ids are hashes of the lex-min
  canonical matrix and never change for the same object. `tests/golden/ids-n4.json`
  pins them and CI enforces it. If a re-keying ever becomes necessary it is a
  breaking change: it gets its own entry here, an alias table for the old ids,
  and advance notice.
- **Response shapes are additive.** New fields may appear; existing fields do
  not change meaning or disappear without an entry here.

## 2026-09-09 — ranks 6, 7 and 8 published; moved to PlanetScale Postgres

### Added

- **Ranks 6, 7 and 8.** The census went from 5,935,362 quivers in 17,735 classes
  to **50,828,164 quivers in 327,961 classes**. Rank 6 is an exact census of its
  cell (|b_ij| ≤ 2); ranks 7 and 8 are documented samples with every
  mutation-finite class explored explicitly rather than left to the sample to
  find. Every mutation-finite class the Felikson–Shapiro–Tumarkin classification
  predicts is present at each rank, including X₆, X₇ and E₆^(1,1).
- **`/api/bulk`** — the whole census as gzipped NDJSON files, one per rank, with
  sha256 checksums and resumable (Range) downloads. Rows are byte-for-byte what
  `/api/export.ndjson` serves.
- **`has_name`** on `/api/quivers` and `/api/classes` — classes carrying a name
  of any kind, automatic Dynkin/surface label or curated nickname (61 classes).
  **`has_nickname`** is the narrower curated set (21).
- **`total_is_lower_bound`** on list responses (see below).
- **`label`** appended to the export columns (`/api/export`, `/api/export.ndjson`
  and the bulk corpus). It is the class's automatic Dynkin/surface name, and
  without it 33 of the 49 named classes — every surface class, 5,254 quivers —
  exported as `dynkin_type: null, nickname: null` and read as unnamed. The three
  name fields are nested: `dynkin_type` (16 classes) ⊂ `label` (49), alongside
  `nickname` (21). Appended, never inserted: CSV consumers index by position.
- **`get_bulk_corpus`** MCP tool.

### Changed

- **Filtered list totals are capped by default.** `total`, `distinct_total` and
  `labeled_total` stop counting at 10,000 when the filter is not rank-only, and
  the response sets `total_is_lower_bound: true`. Pass `?total=exact` for the
  true figure. Counting a filtered cut of a 42.5 M-row rank is a full table
  scan; an unfiltered or rank-only cut is answered from stored aggregates and
  remains exact. **Page with `next_cursor`, not by computing pages from
  `total`.**
- **NULL ordering on nullable sort columns flipped.** Sorting by `class_size`,
  `dynkin_type` or `class_type` now places NULLs last ascending and first
  descending (Postgres's native placement, and the only one an index can serve —
  the alternative cost 60 s per rank-6 page). Non-null ordering is unchanged.
- **Pagination cursors were reissued.** Cursors from before this release are
  rejected with a message naming the change; restart the walk without `?cursor=`.
  Page *order* is unchanged, so nothing is lost but the resume point.
- **Rank-6 responses gained values.** `dynkin_type`, `class_size`, `exploration`
  and `nickname` return real values where 67 % were previously NULL, because
  rank 6's classes are now explored.

### Fixed

- 1,749 rank-8 quivers (0.68 %) referenced a mutation class that was never
  written, so they reported `explored: true` and linked to a class page that
  404ed. Their `mutation_class_id` is now NULL, which is the census's existing
  representation of "not explored". `mutation_finite` is unaffected — that
  verdict came from a real exploration and remains sound.

### Not available at this scale

Some queries cannot be served against a 42.5 M-row rank and now fail fast with
an explanation rather than timing out:

- Sorting a cut larger than 10 M rows by `max_edge`, `class_size`,
  `dynkin_type` or `class_type` — no index can supply that order. The default
  sort and `qmd_id` are indexed and page in milliseconds.
- `?total=exact` on a cut larger than 10 M rows that is not rank-only.

## Earlier

Ranks 1–5 were published on 2026-08-27 (5,935,362 quivers). Before that the
dataset was rank ≤ 4 only. No public changelog was kept; the git history is the
record.
