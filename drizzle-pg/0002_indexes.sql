-- Run AFTER copy + seq population. CONCURRENTLY has no statement timeout,
-- which is the point: CREATE INDEX on 42.5 M rows cannot finish inside D1's
-- 30-second cap, which is why the D1 index set was frozen at load time.
--
-- One index per index in src/db/schema.ts, plus the two seq indexes that
-- replace the SQLite rowid tiebreak.

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

CREATE INDEX CONCURRENTLY idx_lab_qmd_ord             ON labelings (qmd_id, ord);

CREATE UNIQUE INDEX CONCURRENTLY idx_nick_slug        ON class_nicknames (slug);

CREATE INDEX CONCURRENTLY idx_dl_created_at           ON downloads (created_at);
CREATE INDEX CONCURRENTLY idx_dl_email                ON downloads (email);
