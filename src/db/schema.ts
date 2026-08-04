/**
 * Drizzle schema for the QMD D1 database.
 *
 * Mirrors the Postgres schema (qmd/models.py + alembic/) with the naming from
 * the migration brief: `id`, `n`, `exchange_matrix`, `mutation_class_id`,
 * `class_size`. Adapted to SQLite: booleans are INTEGER 0/1, JSON is TEXT.
 *
 * Sharding note: the browse/search tables (`quivers`, `mutation_classes`) stay
 * skinny — small canonical matrices are inlined (a rank-4 matrix is ~50 bytes),
 * but the heavy per-class orbit payloads (every labeled matrix in the class)
 * live in `mutation_class_payloads`, keyed by class id. When per-`n` shards
 * arrive, the skinny tables can serve as a global index DB while the payload
 * table moves into the shards; all access already routes through `shardFor`.
 *
 * Every filterable/sortable column is indexed, and both entity tables carry a
 * composite (n, id) index for the default sort order (`n` first, then id).
 */

import { sql } from "drizzle-orm";
import { index, integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

/** Row-major exchange matrix, e.g. [[0,1],[-1,0]]. */
export type Matrix = number[][];

/** One labeled member of a mutation class's orbit. */
export interface LabeledMember {
  qmd_id: string;
  matrix: Matrix;
}

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

// ---------------------------------------------------------------------------
// mutation_classes — one row per merged mutation class (MC.* id)
// ---------------------------------------------------------------------------

export const mutationClasses = sqliteTable(
  "mutation_classes",
  {
    /** `MC.n{k}.{sha256[:16]}` — the rank is recoverable from the id prefix. */
    id: text("id").primaryKey(),
    n: integer("n").notNull(),
    /** Canonical representative matrix (small; heavy orbits live in payloads). */
    canonicalMatrix: text("canonical_matrix", { mode: "json" })
      .$type<Matrix>()
      .notNull(),
    /** Q.* id of the canonical representative. */
    canonicalQuiverId: text("canonical_quiver_id"),
    /** Mutation bound |b_ij| <= 2 exceeded => class only partially explored. */
    isOpen: integer("is_open", { mode: "boolean" }).notNull(),
    /** Labeled orbit size (explored size for open classes). */
    classSize: integer("class_size").notNull(),
    distinctQuiverCount: integer("distinct_quiver_count").notNull(),
    mergedOrbitCount: integer("merged_orbit_count").notNull().default(1),
    dynkinType: text("dynkin_type"),
    label: text("label"),

    // Finiteness trichotomy (mutually exclusive) + frontier size.
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
    // Default sort order everywhere: n first, then id.
    index("idx_mc_n_id").on(t.n, t.id),
    index("idx_mc_is_open").on(t.isOpen),
    index("idx_mc_dynkin_type").on(t.dynkinType),
    index("idx_mc_class_size").on(t.classSize),
    index("idx_mc_is_mutation_acyclic").on(t.isMutationAcyclic),
    index("idx_mc_is_banff").on(t.isBanff),
    index("idx_mc_is_louise").on(t.isLouise),
    index("idx_mc_is_p_prime").on(t.isPPrime),
  ],
);

// ---------------------------------------------------------------------------
// mutation_class_payloads — heavy per-class JSON, referenced not inlined
// ---------------------------------------------------------------------------

export const mutationClassPayloads = sqliteTable("mutation_class_payloads", {
  mutationClassId: text("mutation_class_id")
    .primaryKey()
    .references(() => mutationClasses.id, { onDelete: "cascade" }),
  /** Full labeled orbit: [{qmd_id, matrix}, ...]. */
  labeledQuivers: text("labeled_quivers", { mode: "json" })
    .$type<LabeledMember[]>()
    .notNull(),
  /** Boundary (frontier) matrices of a partially explored class. */
  boundaryQuivers: text("boundary_quivers", { mode: "json" })
    .$type<Matrix[]>()
    .notNull(),
});

// ---------------------------------------------------------------------------
// quivers — one row per unlabeled quiver isomorphism class (Q.* id)
// ---------------------------------------------------------------------------

export const quivers = sqliteTable(
  "quivers",
  {
    /** `Q.n{k}.{sha256[:16]}` — the rank is recoverable from the id prefix. */
    id: text("id").primaryKey(),
    n: integer("n").notNull(),
    /** Canonical form, row-major JSON. */
    exchangeMatrix: text("exchange_matrix", { mode: "json" })
      .$type<Matrix>()
      .notNull(),
    mutationClassId: text("mutation_class_id").references(
      () => mutationClasses.id,
      { onDelete: "set null" },
    ),

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
    /** How many labeled matrices in the class map to this unlabeled quiver. */
    labelingCount: integer("labeling_count"),
    /** 'finite' / 'tame' / 'wild'; NULL = n/a (cyclic). */
    representationType: text("representation_type"),
    symmetryGroup: text("symmetry_group", { mode: "json" }).$type<SymmetryGroup>(),
  },
  (t) => [
    // Default sort order everywhere: n first, then id.
    index("idx_q_n_id").on(t.n, t.id),
    index("idx_q_mutation_class_id").on(t.mutationClassId),
    index("idx_q_max_edge").on(t.maxEdge),
    index("idx_q_is_acyclic").on(t.isAcyclic),
    index("idx_q_is_connected").on(t.isConnected),
    index("idx_q_is_bipartite").on(t.isBipartite),
    index("idx_q_representation_type").on(t.representationType),
  ],
);

// ---------------------------------------------------------------------------
// rank_stats — aggregates written at ingest time (no query-time scans)
// ---------------------------------------------------------------------------

export const rankStats = sqliteTable("rank_stats", {
  n: integer("n").primaryKey(),
  /** Distinct unlabeled quivers of this rank. */
  quiverCount: integer("quiver_count").notNull(),
  /** Total labeled matrices across this rank (SUM of labeling_count). */
  labeledQuiverCount: integer("labeled_quiver_count").notNull(),
  /** Mutation classes of this rank. */
  classCount: integer("class_count").notNull(),
});

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
    /** 'csv' (xlsx is generated client-side from CSV). */
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
