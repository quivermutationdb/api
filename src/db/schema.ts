/**
 * Drizzle schema for the QMD D1 databases (schema v3, census).
 *
 * Design rules (docs/PHASE2.md §2, docs/PHASE3.md):
 *  - Browse tables (`quivers`, `mutation_classes`) stay skinny. Matrices are
 *    stored in the compact upper-triangular form (src/db/matrix.ts), not JSON.
 *  - Every labeled exchange matrix is its own row in `labelings` — written
 *    only for completely explored (mutation-finite) classes.
 *  - Indexes are rowid-based: rows are inserted in id order per rank, so
 *    `(n, rowid)` IS id order, and keyset cursors use rowid as the tiebreak.
 *    That keeps every secondary index ~4x smaller than one carrying the id.
 *  - Ranks may be split across several databases (data/shards.json); every
 *    database has the same schema, and the global tables (rank_stats,
 *    class_nicknames, downloads) are only populated in the main one.
 *  - `class_nicknames` is curated data (data/nicknames.json) and is never
 *    touched by per-rank imports.
 */

import { sql } from "drizzle-orm";
import { index, integer, primaryKey, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";

/** Row-major exchange matrix, e.g. [[0,1],[-1,0]]. */
export type Matrix = number[][];

/** Per-property provenance for the semidecidable class properties. */
export interface ClassProvenance {
  [property: string]: { state?: string; witness?: unknown; method?: string };
}

/** {order, name, generators} of a quiver's symmetry group. */
export interface SymmetryGroup {
  order?: number;
  name?: string;
  generators?: unknown;
}

/**
 * How far the BFS got:
 *  - complete:  the bounded class was drained — finite, class_size exact
 *  - bound:     a mutation crossed |b_ij| <= bound; for rank >= 3 this proves
 *               the class mutation-infinite (Derksen–Owen)
 *  - truncated: the node cap stopped the search — finiteness UNKNOWN
 */
export type Exploration = "complete" | "bound" | "truncated";

// ---------------------------------------------------------------------------
// mutation_classes — one row per merged mutation class (MC.* id)
// ---------------------------------------------------------------------------

export const mutationClasses = sqliteTable(
  "mutation_classes",
  {
    /** `MC.n{k}.{sha256[:16]}` — the rank is recoverable from the id prefix. */
    id: text("id").primaryKey(),
    n: integer("n").notNull(),
    /** Canonical representative matrix (lex-min over the whole orbit), upper-triangular encoding. */
    canonicalMatrix: text("canonical_matrix").notNull(),
    /** Q.* id of the canonical representative. */
    canonicalQuiverId: text("canonical_quiver_id"),
    /** Compatibility flag: exploration != 'complete'. */
    isOpen: integer("is_open", { mode: "boolean" }).notNull(),
    exploration: text("exploration").$type<Exploration>().notNull().default("complete"),
    /** Labeled orbit size; NULL when the labeled orbit was not computed (only small complete classes get one). */
    classSize: integer("class_size"),
    distinctQuiverCount: integer("distinct_quiver_count").notNull(),
    mergedOrbitCount: integer("merged_orbit_count").notNull().default(1),
    dynkinType: text("dynkin_type"),
    label: text("label"),

    // Finiteness trichotomy (mutually exclusive; all NULL when truncated).
    isFiniteConfirmed: integer("is_finite_confirmed", { mode: "boolean" }),
    isInfiniteConfirmed: integer("is_infinite_confirmed", { mode: "boolean" }),
    isInfiniteExpected: integer("is_infinite_expected", { mode: "boolean" }),
    sizeOfExploredFrontier: integer("size_of_explored_frontier"),

    // Three-state (1 / 0 / NULL = unknown) class properties.
    isMutationAcyclic: integer("is_mutation_acyclic", { mode: "boolean" }),
    isBanff: integer("is_banff", { mode: "boolean" }),
    isLouise: integer("is_louise", { mode: "boolean" }),
    isPPrime: integer("is_p_prime", { mode: "boolean" }),

    /** Per-property method / witness for the semidecidable properties. */
    provenance: text("provenance", { mode: "json" }).$type<ClassProvenance>(),
  },
  (t) => [
    // rowid-based (rows are inserted in id order per rank): (n, rowid) = id order.
    index("idx_mc_n").on(t.n),
    index("idx_mc_n_class_size").on(t.n, t.classSize),
    index("idx_mc_n_dynkin").on(t.n, t.dynkinType),
    index("idx_mc_n_distinct").on(t.n, t.distinctQuiverCount),
    index("idx_mc_n_open").on(t.n, t.isOpen),
    index("idx_mc_finite_confirmed").on(t.isFiniteConfirmed),
    index("idx_mc_infinite_confirmed").on(t.isInfiniteConfirmed),
    index("idx_mc_is_mutation_acyclic").on(t.isMutationAcyclic),
  ],
);

// ---------------------------------------------------------------------------
// labelings — one row per labeled exchange matrix in a class's orbit
// ---------------------------------------------------------------------------

export const labelings = sqliteTable(
  "labelings",
  {
    mutationClassId: text("mutation_class_id")
      .notNull()
      .references(() => mutationClasses.id, { onDelete: "cascade" }),
    /** Position in the class's lex-sorted orbit (deterministic). */
    ord: integer("ord").notNull(),
    /** Unlabeled quiver this labeling belongs to. */
    qmdId: text("qmd_id").notNull(),
    /** Upper-triangular encoding (src/db/matrix.ts). */
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

export const quivers = sqliteTable(
  "quivers",
  {
    /** `Q.n{k}.{sha256[:16]}` — the rank is recoverable from the id prefix. */
    id: text("id").primaryKey(),
    n: integer("n").notNull(),
    /** Canonical form (lex-min), upper-triangular encoding (src/db/matrix.ts). */
    exchangeMatrix: text("exchange_matrix").notNull(),
    /**
     * NULL = the class of this quiver was not explored (quiver-only row). No
     * foreign key: for split ranks the class row may live in another shard.
     */
    mutationClassId: text("mutation_class_id"),
    /**
     * Three-state mutation-finiteness of the quiver's class, known even when
     * no class row exists: 0 = proved infinite (Derksen–Owen: an entry
     * |b_ij| >= 3 at rank >= 3, found in the quiver itself or by exploring),
     * 1 = proved finite (class explored completely), NULL = unknown.
     */
    mutationFinite: integer("mutation_finite", { mode: "boolean" }),

    // Per-quiver invariants (stored so the API can filter on them).
    maxEdge: integer("max_edge").notNull().default(0),
    isAcyclic: integer("is_acyclic", { mode: "boolean" }).notNull().default(true),
    isConnected: integer("is_connected", { mode: "boolean" })
      .notNull()
      .default(true),
    isBipartite: integer("is_bipartite", { mode: "boolean" }),
    isAbundant: integer("is_abundant", { mode: "boolean" }),
    /** NULL = unknown (n > 4). */
    isPlanar: integer("is_planar", { mode: "boolean" }),
    /** How many labeled matrices in the explored class map to this unlabeled quiver. */
    labelingCount: integer("labeling_count"),
    /** 'finite' / 'tame' / 'wild'; NULL = n/a (cyclic). */
    representationType: text("representation_type"),
    symmetryGroup: text("symmetry_group", { mode: "json" }).$type<SymmetryGroup>(),
  },
  (t) => [
    // rowid-based: (n, rowid) is id order because rows are inserted sorted by id.
    index("idx_q_n").on(t.n),
    index("idx_q_n_max_edge").on(t.n, t.maxEdge),
    index("idx_q_n_finite").on(t.n, t.mutationFinite),
    index("idx_q_representation_type").on(t.representationType),
    // Class members ordered by prominence (most labelings first).
    index("idx_q_mc_labcount").on(t.mutationClassId, t.labelingCount),
  ],
);

// ---------------------------------------------------------------------------
// rank_stats — aggregates + provenance written at ingest time
// ---------------------------------------------------------------------------

export const rankStats = sqliteTable("rank_stats", {
  n: integer("n").primaryKey(),
  /** Distinct unlabeled quivers of this rank. */
  quiverCount: integer("quiver_count").notNull(),
  /** Total labeled matrices across this rank (SUM of labeling_count). */
  labeledQuiverCount: integer("labeled_quiver_count").notNull(),
  /** Mutation classes of this rank. */
  classCount: integer("class_count").notNull(),
  /** How this rank was generated. */
  bound: integer("bound"),
  nodeCap: integer("node_cap"),
  generatedAt: text("generated_at"),
  pipelineVersion: text("pipeline_version"),
  /** 'orderly' (exact census) | 'brute' | 'sample'. */
  generator: text("generator"),
  /** Exact number of connected unlabeled quivers in the cell (n, bound). */
  censusSize: integer("census_size"),
  /** Per-shard row counts for split ranks: {shardKey: {quivers, classes}}. */
  shardCounts: text("shard_counts", { mode: "json" }).$type<Record<string, { quivers: number; classes: number }>>(),
});

// ---------------------------------------------------------------------------
// class_nicknames — curated names (data/nicknames.json); survives re-imports
// ---------------------------------------------------------------------------

export const classNicknames = sqliteTable(
  "class_nicknames",
  {
    mcId: text("mc_id").primaryKey(),
    nickname: text("nickname").notNull(),
    /** URL-safe key: /class.html?name={slug}, /api/classes/by-slug/{slug}. */
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

export const downloads = sqliteTable(
  "downloads",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    /** ISO 8601 UTC ("YYYY-MM-DD HH:MM:SS"). */
    createdAt: text("created_at")
      .notNull()
      .default(sql`CURRENT_TIMESTAMP`),
    /** 'csv' | 'ndjson' (xlsx is generated client-side from CSV). */
    fmt: text("fmt").notNull(),
    rowCount: integer("row_count").notNull(),
    /** The applied cut (non-null filters). */
    filters: text("filters", { mode: "json" }).$type<Record<string, unknown>>(),
    /** Optional, self-reported. */
    email: text("email"),
    /** Optional name / affiliation. */
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
