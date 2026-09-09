/**
 * Drizzle schema for QMD on PostgreSQL (schema v3, census) — the port target.
 *
 * This mirrors src/db/schema.ts field for field. It exists in parallel because
 * the live Worker still runs on D1; at cutover, imports move here and the
 * sqlite-core file is deleted. Until then, a change to one must be made to the
 * other — there is no generator keeping them in step.
 *
 * `drizzle-pg/*.sql` is AUTHORITATIVE for DDL, not this file. The hand-written
 * DDL deliberately withholds primary keys from the three bulk-loaded tables so
 * the COPY hits a bare heap (a primary key is an index and bloats like one:
 * measured 1.67x on quivers_pkey, 1.40 GB reclaimed by deferring). Drizzle
 * declares the keys here because the query builder needs to know about them;
 * do NOT run drizzle-kit generate against this file and apply the result.
 *
 * Differences from the SQLite declaration, all measured or decided in the
 * pre-flight review (docs/PLANETSCALE.md):
 *
 *  - `seq integer` replaces the SQLite rowid as the keyset tiebreak. Assigned
 *    at load by row_number() OVER (PARTITION BY n ORDER BY id), so (n, seq) IS
 *    id order — the property src/api/cursor.ts relies on. INTEGER, not bigint:
 *    the largest rank is 42.5 M rows and the largest counter anywhere is
 *    573 M, both inside int4. Measured, idx_q_n_seq costs 1,089 MB against
 *    quivers_pkey's 1,969 MB, so an int4 seq index is about half the price of
 *    a text-id one.
 *  - Three-state integers become real `boolean` (true / false / NULL).
 *  - JSON text becomes `jsonb`. Read-side safe (only ever accessed as parsed
 *    objects), but jsonb does not preserve key order, so response bytes change
 *    where `provenance` is passed through whole.
 *  - rank_stats counters are `bigint` for forward safety — a rank-9 cell would
 *    exceed int4. Declared mode:"number" so the TS types are unchanged;
 *    Postgres returns bigint as a string, which is also why the aggregate call
 *    sites coerce with Number() (see src/api/quivers.ts totalsFor).
 *  - `downloads.created_at` becomes `timestamptz`. Safe: the table is
 *    insert-only and nothing in src/api reads it back.
 *  - id columns are COLLATE "C" in the DDL so text comparison is byte order,
 *    matching SQLite, the documented id ordering, and JS. Collation is not
 *    expressible in the Drizzle column builder; it lives in drizzle-pg only.
 */

import {
  bigint, boolean, index, integer, jsonb, pgTable, primaryKey, text,
  timestamp, uniqueIndex,
} from "drizzle-orm/pg-core";

export type { ClassProvenance, Exploration, Matrix, SymmetryGroup } from "./schema";
import type { ClassProvenance, Exploration, SymmetryGroup } from "./schema";

// ---------------------------------------------------------------------------
// mutation_classes — one row per merged mutation class (MC.* id)
// ---------------------------------------------------------------------------

export const mutationClasses = pgTable(
  "mutation_classes",
  {
    id: text("id").primaryKey(),
    n: integer("n").notNull(),
    canonicalMatrix: text("canonical_matrix").notNull(),
    canonicalQuiverId: text("canonical_quiver_id"),
    isOpen: boolean("is_open").notNull(),
    exploration: text("exploration").$type<Exploration>().notNull().default("complete"),
    classSize: integer("class_size"),
    distinctQuiverCount: integer("distinct_quiver_count").notNull(),
    mergedOrbitCount: integer("merged_orbit_count").notNull().default(1),
    dynkinType: text("dynkin_type"),
    label: text("label"),

    isFiniteConfirmed: boolean("is_finite_confirmed"),
    isInfiniteConfirmed: boolean("is_infinite_confirmed"),
    isInfiniteExpected: boolean("is_infinite_expected"),
    sizeOfExploredFrontier: integer("size_of_explored_frontier"),

    isMutationAcyclic: boolean("is_mutation_acyclic"),
    isBanff: boolean("is_banff"),
    isLouise: boolean("is_louise"),
    isPPrime: boolean("is_p_prime"),

    provenance: jsonb("provenance").$type<ClassProvenance>(),

    /** Keyset tiebreak; (n, seq) is id order. Replaces the SQLite rowid. */
    seq: integer("seq"),
  },
  (t) => [
    index("idx_mc_n").on(t.n),
    index("idx_mc_n_class_size").on(t.n, t.classSize),
    index("idx_mc_n_dynkin").on(t.n, t.dynkinType),
    index("idx_mc_n_distinct").on(t.n, t.distinctQuiverCount),
    index("idx_mc_n_open").on(t.n, t.isOpen),
    index("idx_mc_finite_confirmed").on(t.isFiniteConfirmed),
    index("idx_mc_infinite_confirmed").on(t.isInfiniteConfirmed),
    index("idx_mc_is_mutation_acyclic").on(t.isMutationAcyclic),
    index("idx_mc_n_seq").on(t.n, t.seq),
  ],
);

// ---------------------------------------------------------------------------
// labelings — one row per labeled exchange matrix in a class's orbit
// ---------------------------------------------------------------------------

export const labelings = pgTable(
  "labelings",
  {
    mutationClassId: text("mutation_class_id")
      .notNull()
      .references(() => mutationClasses.id, { onDelete: "cascade" }),
    ord: integer("ord").notNull(),
    qmdId: text("qmd_id").notNull(),
    matrix: text("matrix").notNull(),
  },
  (t) => [
    primaryKey({ columns: [t.mutationClassId, t.ord] }),
    index("idx_lab_qmd_ord").on(t.qmdId, t.ord),
  ],
);

// ---------------------------------------------------------------------------
// quivers — one row per unlabeled quiver isomorphism class (Q.* id)
// ---------------------------------------------------------------------------

export const quivers = pgTable(
  "quivers",
  {
    id: text("id").primaryKey(),
    n: integer("n").notNull(),
    exchangeMatrix: text("exchange_matrix").notNull(),
    /** No foreign key: a quiver may have no explored class. */
    mutationClassId: text("mutation_class_id"),
    mutationFinite: boolean("mutation_finite"),

    maxEdge: integer("max_edge").notNull().default(0),
    isAcyclic: boolean("is_acyclic").notNull().default(true),
    isConnected: boolean("is_connected").notNull().default(true),
    isBipartite: boolean("is_bipartite"),
    isAbundant: boolean("is_abundant"),
    /** NULL = unknown (n > 4). */
    isPlanar: boolean("is_planar"),
    labelingCount: integer("labeling_count"),
    representationType: text("representation_type"),
    symmetryGroup: jsonb("symmetry_group").$type<SymmetryGroup>(),

    /** Keyset tiebreak; (n, seq) is id order. Replaces the SQLite rowid. */
    seq: integer("seq"),
  },
  (t) => [
    index("idx_q_n").on(t.n),
    index("idx_q_n_max_edge").on(t.n, t.maxEdge),
    index("idx_q_n_finite").on(t.n, t.mutationFinite),
    index("idx_q_representation_type").on(t.representationType),
    index("idx_q_mc_labcount").on(t.mutationClassId, t.labelingCount),
    index("idx_q_n_seq").on(t.n, t.seq),
    // Three partial indexes also exist on this table -- idx_q_n_acyclic,
    // idx_q_n_disconn, idx_q_n_explored -- declared only in
    // drizzle-pg/0002_indexes.sql. They are planner-only (they cover the rare
    // side of a lopsided boolean filter; the capped count covers the common
    // side), the query builder never names them, and Drizzle has no partial
    // index builder, so declaring them here would just make drizzle-kit want
    // to drop them. Delete them there, not here.
  ],
);

// ---------------------------------------------------------------------------
// rank_stats — aggregates + provenance written at ingest time
// ---------------------------------------------------------------------------

export const rankStats = pgTable("rank_stats", {
  n: integer("n").primaryKey(),
  /** bigint: forward safety for a rank-9 cell. Coerce reads with Number(). */
  quiverCount: bigint("quiver_count", { mode: "number" }).notNull(),
  labeledQuiverCount: bigint("labeled_quiver_count", { mode: "number" }).notNull(),
  classCount: bigint("class_count", { mode: "number" }).notNull(),
  bound: integer("bound"),
  nodeCap: integer("node_cap"),
  generatedAt: text("generated_at"),
  pipelineVersion: text("pipeline_version"),
  generator: text("generator"),
  censusSize: bigint("census_size", { mode: "number" }),
  /**
   * Retained for provenance of how a rank was generated. After the collapse to
   * one database there are no shards to count, so new ranks leave it NULL.
   */
  shardCounts: jsonb("shard_counts").$type<Record<string, { quivers: number; classes: number }>>(),
});

// ---------------------------------------------------------------------------
// class_nicknames — curated names (data/nicknames.json); survives re-imports
// ---------------------------------------------------------------------------

export const classNicknames = pgTable(
  "class_nicknames",
  {
    mcId: text("mc_id").primaryKey(),
    nickname: text("nickname").notNull(),
    slug: text("slug").notNull(),
    note: text("note"),
    addedBy: text("added_by"),
    addedAt: text("added_at"),
  },
  (t) => [uniqueIndex("idx_nick_slug").on(t.slug)],
);

// ---------------------------------------------------------------------------
// downloads — one row per dataset export (usage tracking, no site accounts)
// ---------------------------------------------------------------------------

export const downloads = pgTable(
  "downloads",
  {
    id: bigint("id", { mode: "number" }).primaryKey().generatedAlwaysAsIdentity(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
    fmt: text("fmt").notNull(),
    rowCount: bigint("row_count", { mode: "number" }).notNull(),
    filters: jsonb("filters").$type<Record<string, unknown>>(),
    email: text("email"),
    name: text("name"),
    ip: text("ip"),
    userAgent: text("user_agent"),
    referer: text("referer"),
  },
  (t) => [
    index("idx_dl_created_at").on(t.createdAt),
    index("idx_dl_email").on(t.email),
  ],
);
