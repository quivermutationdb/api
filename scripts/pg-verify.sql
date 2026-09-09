-- STRUCTURAL verification of a Postgres load. Dataset-agnostic on purpose: it
-- asserts things that must hold of ANY correct load, so CI can run it against
-- the small dev cell and a release can run it against the full census.
--   psql -f scripts/pg-verify.sql
-- The census-specific numbers (50,828,164 rows, 19 finite rank-8 classes,
-- E6^(1,1), ...) live in scripts/pg-verify-census.sql -- do not merge the two,
-- or CI will start asserting row counts it cannot possibly have.
-- Everything marked 'must be' is a hard expectation; a false or a nonzero
-- orphan count means the load is wrong, not merely surprising.

\echo '=== row counts by rank ==='
SELECT n, count(*) AS quivers FROM quivers GROUP BY n ORDER BY n;
SELECT count(*) AS total_quivers FROM quivers;
SELECT count(*) AS mutation_classes FROM mutation_classes;
SELECT count(*) AS labelings FROM labelings;
SELECT count(*) AS nicknames FROM class_nicknames;
SELECT count(*) AS rank_stats_rows FROM rank_stats;

\echo '=== seq: unique, gapless 1..count, per rank (all must be t) ==='
SELECT n, count(*) = max(seq) AS ends_at_count, min(seq) = 1 AS starts_at_one,
       count(DISTINCT seq) = count(*) AS unique_seq
FROM quivers GROUP BY n ORDER BY n;

\echo '=== (n, seq) reproduces id order at every rank (all must be t) ==='
SELECT n, bool_and(ok) AS seq_is_id_order FROM (
  SELECT n, seq = row_number() OVER (PARTITION BY n ORDER BY id) AS ok FROM quivers
) x GROUP BY n ORDER BY n;

\echo '=== referential integrity ==='
SELECT count(*) AS orphan_labelings FROM labelings l
  LEFT JOIN mutation_classes m ON l.mutation_class_id = m.id WHERE m.id IS NULL;
SELECT count(*) AS orphan_quiver_class_refs FROM quivers q
  LEFT JOIN mutation_classes m ON q.mutation_class_id = m.id
  WHERE q.mutation_class_id IS NOT NULL AND m.id IS NULL;
SELECT count(*) AS nicknames_without_class FROM class_nicknames c
  LEFT JOIN mutation_classes m ON c.mc_id = m.id WHERE m.id IS NULL;

\echo '=== C collation survived the load (must be 7) ==='
SELECT count(*) AS c_collated_columns FROM information_schema.columns
 WHERE table_schema = 'public' AND collation_name = 'C';

\echo '=== size ==='
SELECT relname, pg_size_pretty(pg_total_relation_size(c.oid)) AS total
  FROM pg_class c JOIN pg_namespace nsp ON nsp.oid = c.relnamespace
 WHERE nsp.nspname = 'public' AND c.relkind = 'r' ORDER BY pg_total_relation_size(c.oid) DESC;
SELECT pg_size_pretty(sum(pg_total_relation_size(c.oid))) AS database_total
  FROM pg_class c JOIN pg_namespace nsp ON nsp.oid = c.relnamespace
 WHERE nsp.nspname = 'public' AND c.relkind = 'r';

\echo '=== rank_stats must agree with the rows actually loaded (all must be t) ==='
SELECT r.n,
       r.quiver_count = q.actual_quivers      AS quiver_count_ok,
       r.class_count  = coalesce(c.actual_classes, 0) AS class_count_ok,
       r.census_size  AS census_size, r.generator
FROM rank_stats r
LEFT JOIN (SELECT n, count(*) actual_quivers FROM quivers GROUP BY n) q ON q.n = r.n
LEFT JOIN (SELECT n, count(*) actual_classes FROM mutation_classes GROUP BY n) c ON c.n = r.n
ORDER BY r.n;

\echo '=== every loaded rank has a rank_stats row (must be 0 missing) ==='
SELECT count(*) AS ranks_without_stats FROM (
  SELECT DISTINCT n FROM quivers EXCEPT SELECT n FROM rank_stats) x;

\echo '=== dangling class references (must be 0; see drizzle-pg/0003) ==='
SELECT count(*) AS dangling_class_refs FROM quivers q
 WHERE q.mutation_class_id IS NOT NULL
   AND NOT EXISTS (SELECT 1 FROM mutation_classes m WHERE m.id = q.mutation_class_id);

\echo '=== no invalid indexes left by CONCURRENTLY (must be none) ==='
SELECT coalesce(string_agg(c.relname, ', '), '(none)') AS invalid_indexes
  FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid WHERE NOT i.indisvalid;
