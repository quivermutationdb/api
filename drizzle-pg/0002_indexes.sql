-- Run AFTER copy + seq population. CONCURRENTLY has no statement timeout,
-- which is the point: CREATE INDEX on 42.5 M rows cannot finish inside D1's
-- 30-second cap, which is why the D1 index set was frozen at load time.
--
-- One index per index in src/db/schema.ts, plus the two seq indexes that
-- replace the SQLite rowid tiebreak, plus the primary keys and the foreign
-- key that 0001 deliberately withheld so the bulk load hits a bare heap.
--
-- Primary keys are added as CREATE UNIQUE INDEX CONCURRENTLY followed by
-- ALTER TABLE ... ADD PRIMARY KEY USING INDEX, which avoids holding an
-- exclusive lock for the length of a 50 M-row index build.

CREATE UNIQUE INDEX CONCURRENTLY quivers_pkey          ON quivers (id);
ALTER TABLE quivers          ADD PRIMARY KEY USING INDEX quivers_pkey;
CREATE UNIQUE INDEX CONCURRENTLY mutation_classes_pkey ON mutation_classes (id);
ALTER TABLE mutation_classes ADD PRIMARY KEY USING INDEX mutation_classes_pkey;
CREATE UNIQUE INDEX CONCURRENTLY labelings_pkey        ON labelings (mutation_class_id, ord);
ALTER TABLE labelings        ADD PRIMARY KEY USING INDEX labelings_pkey;

-- Validated separately so the scan does not block the table.
ALTER TABLE labelings ADD CONSTRAINT labelings_mutation_class_id_fkey
  FOREIGN KEY (mutation_class_id) REFERENCES mutation_classes(id)
  ON DELETE CASCADE NOT VALID;
ALTER TABLE labelings VALIDATE CONSTRAINT labelings_mutation_class_id_fkey;

CREATE INDEX CONCURRENTLY idx_mc_n                    ON mutation_classes (n);
CREATE INDEX CONCURRENTLY idx_mc_n_class_size         ON mutation_classes (n, class_size);
CREATE INDEX CONCURRENTLY idx_mc_n_dynkin             ON mutation_classes (n, dynkin_type);
CREATE INDEX CONCURRENTLY idx_mc_n_distinct           ON mutation_classes (n, distinct_quiver_count);
CREATE INDEX CONCURRENTLY idx_mc_n_open               ON mutation_classes (n, is_open);
CREATE INDEX CONCURRENTLY idx_mc_finite_confirmed     ON mutation_classes (is_finite_confirmed);
CREATE INDEX CONCURRENTLY idx_mc_infinite_confirmed   ON mutation_classes (is_infinite_confirmed);
CREATE INDEX CONCURRENTLY idx_mc_is_mutation_acyclic  ON mutation_classes (is_mutation_acyclic);
CREATE INDEX CONCURRENTLY idx_mc_n_seq                ON mutation_classes (n, seq);

CREATE INDEX CONCURRENTLY idx_q_n                     ON quivers (n);
CREATE INDEX CONCURRENTLY idx_q_n_max_edge            ON quivers (n, max_edge);
CREATE INDEX CONCURRENTLY idx_q_n_finite              ON quivers (n, mutation_finite);
CREATE INDEX CONCURRENTLY idx_q_representation_type   ON quivers (representation_type);
CREATE INDEX CONCURRENTLY idx_q_mc_labcount           ON quivers (mutation_class_id, labeling_count);
CREATE INDEX CONCURRENTLY idx_q_n_seq                 ON quivers (n, seq);

-- Partial indexes for the filterable quiver columns that have no (n, col)
-- index of their own: is_acyclic, is_connected and explored (mutation_class_id
-- IS NOT NULL). Each covers only the RARE side of a lopsided boolean; the
-- common side needs no index, because the API's capped count (TOTAL_CAP in
-- src/api/quivers.ts) stops as soon as it has seen enough rows. The two
-- together remove the last query shape that scanned the 9 GB heap.
--
-- Measured on the Phase 0 load, single-threaded, cap 10k (docs/PLANETSCALE.md):
--   n=6 AND NOT is_connected     5956 ms -> 0.045 ms   (index is 8 kB: no such row)
--   n=6 AND is_acyclic (exact)   6002 ms -> 413 ms
--   n=6 AND explored (exact)        --   -> 198 ms
-- Total cost of all three: 69 MB, against 5.07 GB of existing index.
CREATE INDEX CONCURRENTLY idx_q_n_acyclic   ON quivers (n) WHERE is_acyclic;
CREATE INDEX CONCURRENTLY idx_q_n_disconn   ON quivers (n) WHERE NOT is_connected;
CREATE INDEX CONCURRENTLY idx_q_n_explored  ON quivers (n) WHERE mutation_class_id IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_lab_qmd_ord             ON labelings (qmd_id, ord);

CREATE UNIQUE INDEX CONCURRENTLY idx_nick_slug        ON class_nicknames (slug);

CREATE INDEX CONCURRENTLY idx_dl_created_at           ON downloads (created_at);
CREATE INDEX CONCURRENTLY idx_dl_email                ON downloads (email);


-- ---------------------------------------------------------------------------
-- Planner settings
-- ---------------------------------------------------------------------------
-- random_page_cost defaults to 4.0, a ratio calibrated for seek-bound spinning
-- disks. On SSD the true ratio is near 1, and the stale default makes the
-- planner reject perfectly good index scans in favour of scanning the whole
-- 9 GB quivers heap. Measured, single-threaded: `n = 7 AND is_acyclic` went
-- from a 5.5 s Seq Scan to a 0.65 s Index Scan on idx_q_n from this one
-- setting -- 8.4x, no schema change. Rank 6 still (correctly) prefers a scan,
-- being 84 % of the table.
--
-- Wrapped because a managed provider may not grant ALTER DATABASE; if it does
-- not, set it in the PlanetScale console instead, or the Worker will silently
-- run every filtered count against the spinning-disk cost model.
DO $$
BEGIN
  EXECUTE format('ALTER DATABASE %I SET random_page_cost = 1.1', current_database());
EXCEPTION WHEN insufficient_privilege THEN
  RAISE NOTICE 'could not ALTER DATABASE: set random_page_cost = 1.1 in the PlanetScale console';
END $$;

ANALYZE;
