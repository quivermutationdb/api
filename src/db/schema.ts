/**
 * Drizzle schema for the QMD D1 database (schema v2, phase 2).
 *
 * Design rules (docs/PHASE2.md §2):
 *  - Browse tables (`quivers`, `mutation_classes`) stay skinny: a canonical
 *    matrix is inlined (a rank-10 matrix is ~400 bytes), nothing else heavy.
 *  - Every labeled exchange matrix is its own row in `labelings`; every
 *    frontier matrix its own row in `frontier_quivers`. D1 caps a row at 2 MB
 *    and a D_10 orbit alone is ~40 MB of JSON, so orbits are never blobs.
 *  - Every list the API serves is index-ordered and keyset-pageable; the
 *    composite indexes below are exactly the (filter, sort, id) shapes used.
 *  - `class_nicknames` is curated data (data/nicknames.json) and is never
 *    touched by per-rank imports.
 *
 * All access goes through `shardFor` (src/db/shard.ts); ids encode the rank.
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
    /** Canonical representative matrix (lex-min over the whole orbit). */
    canonicalMatrix: text("canonical_matrix", { mode: "json" })
      .$type<Matrix>()
      .notNull(),
    /** Q.* id of the canonical representative. */
    canonicalQuiverId: text("canonical_quiver_id"),
    /** Compatibility flag: exploration != 'complete'. */
    isOpen: integer("is_open", { mode: "boolean" }).notNull(),
    exploration: text("exploration").$type<Exploration>().notNull().default("complete"),
    /** Labeled orbit size (explored size unless exploration = complete). */
    classSize: integer("class_size").notNull(),
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
    // Default order everywhere: n first, then id.
    index("idx_mc_n_id").on(t.n, t.id),
    // Sortable columns, each with the (n, col, id) shape the API queries use.
    index("idx_mc_n_class_size_id").on(t.n, t.classSize, t.id),
    index("idx_mc_n_dynkin_id").on(t.n, t.dynkinType, t.id),
    index("idx_mc_n_distinct_id").on(t.n, t.distinctQuiverCount, t.id),
    index("idx_mc_n_open_id").on(t.n, t.isOpen, t.id),
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
    matrix: text("matrix", { mode: "json" }).$type<Matrix>().notNull(),
  },
  (t) => [
    primaryKey({ columns: [t.mutationClassId, t.ord] }),
    index("idx_lab_qmd_ord").on(t.qmdId, t.ord),
    index("idx_lab_mc_qmd_ord").on(t.mutationClassId, t.qmdId, t.ord),
  ],
);

// ---------------------------------------------------------------------------
// frontier_quivers — matrices with an escaping (bound-crossing) mutation
// ---------------------------------------------------------------------------

export const frontierQuivers = sqliteTable(
  "frontier_quivers",
  {
    mutationClassId: text("mutation_class_id")
      .notNull()
      .references(() => mutationClasses.id, { onDelete: "cascade" }),
    ord: integer("ord").notNull(),
    matrix: text("matrix", { mode: "json" }).$type<Matrix>().notNull(),
  },
  (t) => [primaryKey({ columns: [t.mutationClassId, t.ord] })],
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
    /** Canonical form (lex-min), row-major JSON. */
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
    /** Prefix sum of labeling_count over this rank in id order (labelings windowing). */
    labelingOffset: integer("labeling_offset"),
    /** 'finite' / 'tame' / 'wild'; NULL = n/a (cyclic). */
    representationType: text("representation_type"),
    symmetryGroup: text("symmetry_group", { mode: "json" }).$type<SymmetryGroup>(),
  },
  (t) => [
    index("idx_q_n_id").on(t.n, t.id),
    index("idx_q_n_max_edge_id").on(t.n, t.maxEdge, t.id),
    index("idx_q_n_labeling_offset").on(t.n, t.labelingOffset),
    index("idx_q_representation_type").on(t.representationType),
    // Class members ordered by prominence (most labelings first), then id.
    index("idx_q_mc_labcount_id").on(t.mutationClassId, t.labelingCount, t.id),
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
  /** Exact number of unlabeled quivers in the cell (n, bound) — Burnside count. */
  censusSize: integer("census_size"),
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
