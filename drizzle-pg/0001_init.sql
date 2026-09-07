-- QMD schema v3, PostgreSQL translation of src/db/schema.ts.
--
-- Differences from the D1/SQLite schema, all decided in the pre-flight review
-- (docs/PLANETSCALE.md):
--
--   * id columns are COLLATE "C". Postgres' default collation is not byte
--     order; C restores it, which matches SQLite, matches the documented id
--     ordering, keeps JS string comparison in agreement, and keeps the index
--     usable by the keyset predicates in src/api/cursor.ts.
--   * `seq integer` replaces the SQLite rowid used as the keyset tiebreak.
--     INTEGER, not bigint: the largest rank holds 42.5 M rows and the largest
--     counter anywhere is 573 M, both inside int4. Half the index width, and
--     it keeps bigint-as-string out of cursor keys, where compareKeys compares
--     numerically.
--   * mode:"boolean" integers become real `boolean` (three-state preserved:
--     true / false / NULL = unknown).
--   * mode:"json" text becomes `jsonb`. Read-side safe (only ever accessed as
--     parsed objects), but jsonb does not preserve key order, so response
--     bytes change where provenance is passed through whole.
--   * rank_stats counters are bigint for forward safety (a rank-9 cell would
--     exceed int4). Coerce with Number() on read -- Postgres returns bigint as
--     a string.
--
-- Indexes are NOT created here. Load with COPY first, then populate seq, then
-- CREATE INDEX CONCURRENTLY (0002_indexes.sql). That ordering is one of the
-- four reasons for leaving D1.

CREATE TABLE mutation_classes (
  id                          text COLLATE "C" PRIMARY KEY,
  n                           integer NOT NULL,
  canonical_matrix            text    NOT NULL,
  canonical_quiver_id         text COLLATE "C",
  is_open                     boolean NOT NULL,
  exploration                 text    NOT NULL DEFAULT 'complete',
  class_size                  integer,
  distinct_quiver_count       integer NOT NULL,
  merged_orbit_count          integer NOT NULL DEFAULT 1,
  dynkin_type                 text,
  label                       text,
  is_finite_confirmed         boolean,
  is_infinite_confirmed       boolean,
  is_infinite_expected        boolean,
  size_of_explored_frontier   integer,
  is_mutation_acyclic         boolean,
  is_banff                    boolean,
  is_louise                   boolean,
  is_p_prime                  boolean,
  provenance                  jsonb,
  seq                         integer,
  CONSTRAINT mutation_classes_exploration_check
    CHECK (exploration IN ('complete', 'bound', 'truncated'))
);

CREATE TABLE quivers (
  id                    text COLLATE "C" PRIMARY KEY,
  n                     integer NOT NULL,
  exchange_matrix       text    NOT NULL,
  mutation_class_id     text COLLATE "C",          -- no FK: see schema.ts
  mutation_finite       boolean,
  max_edge              integer NOT NULL DEFAULT 0,
  is_acyclic            boolean NOT NULL DEFAULT true,
  is_connected          boolean NOT NULL DEFAULT true,
  is_bipartite          boolean,
  is_abundant           boolean,
  is_planar             boolean,
  labeling_count        integer,
  representation_type   text,
  symmetry_group        jsonb,
  seq                   integer
);

CREATE TABLE labelings (
  mutation_class_id text COLLATE "C" NOT NULL
    REFERENCES mutation_classes(id) ON DELETE CASCADE,
  ord               integer NOT NULL,
  qmd_id            text COLLATE "C" NOT NULL,
  matrix            text    NOT NULL,
  PRIMARY KEY (mutation_class_id, ord)
);

CREATE TABLE rank_stats (
  n                     integer PRIMARY KEY,
  quiver_count          bigint  NOT NULL,
  labeled_quiver_count  bigint  NOT NULL,
  class_count           bigint  NOT NULL,
  bound                 integer,
  node_cap              integer,
  generated_at          text,
  pipeline_version      text,
  generator             text,
  census_size           bigint,
  shard_counts          jsonb
);

CREATE TABLE class_nicknames (
  mc_id     text COLLATE "C" PRIMARY KEY,
  nickname  text NOT NULL,
  slug      text NOT NULL,
  note      text,
  added_by  text,
  added_at  text
);

CREATE TABLE downloads (
  id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  created_at  timestamptz NOT NULL DEFAULT now(),
  fmt         text        NOT NULL,
  row_count   bigint      NOT NULL,
  filters     jsonb,
  email       text,
  name        text,
  ip          text,
  user_agent  text,
  referer     text
);
