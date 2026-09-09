-- CENSUS verification: the specific numbers of the published dataset.
-- Run after scripts/pg-verify.sql on a FULL load only; every figure here is a
-- fact about the real census, so it fails by design on the dev cell.
--   psql -f scripts/pg-verify-census.sql
--
-- These are the mathematically load-bearing rows. A load can be structurally
-- perfect -- gapless seq, clean foreign keys -- and still have quietly dropped
-- the 428 mutation-finite quivers at rank 6 or the one E6^(1,1) class at rank
-- 8. Those are the rows the database exists for.

\echo '=== totals (must be exact) ==='
SELECT count(*) = 50828164 AS quivers_ok       FROM quivers;
SELECT count(*) = 327961   AS classes_ok       FROM mutation_classes;
SELECT count(*) = 457125   AS labelings_ok     FROM labelings;
SELECT count(*) = 8        AS all_eight_ranks  FROM rank_stats;

\echo '=== the mathematically load-bearing rows ==='
SELECT 'rank-8 finite classes (must be 19)' AS check, count(*)::text AS value
  FROM mutation_classes WHERE n = 8 AND is_finite_confirmed;
SELECT 'E6^(1,1) present, 49 quivers', coalesce(distinct_quiver_count::text, 'MISSING')
  FROM mutation_classes WHERE id = 'MC.n8.e9303af7e4328df4';
SELECT 'rank-6 mutation-finite quivers (must be 428)', count(*)::text
  FROM quivers WHERE n = 6 AND mutation_finite;
SELECT 'rank-6 finite classes (must be 13)', count(*)::text
  FROM mutation_classes WHERE n = 6 AND is_finite_confirmed;
SELECT 'Markov not mutation-acyclic', is_mutation_acyclic::text
  FROM mutation_classes WHERE id = 'MC.n3.7405511b230b7552';


\echo '=== per-rank quiver counts (must all be t) ==='
SELECT n, count(*) = expected AS ok, count(*) AS actual, expected
FROM quivers JOIN (VALUES (1,1),(2,10),(3,1550),(4,3574495),(5,2359306),
                          (6,42514454),(7,2120315),(8,258033)) AS e(rn, expected)
  ON e.rn = quivers.n
GROUP BY n, expected ORDER BY n;

\echo '=== every curated nickname resolves to a real class (must be 0) ==='
SELECT count(*) AS nicknames_without_class FROM class_nicknames c
  LEFT JOIN mutation_classes m ON c.mc_id = m.id WHERE m.id IS NULL;
