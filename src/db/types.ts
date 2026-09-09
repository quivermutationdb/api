/**
 * Domain types shared by both schema files and by the API.
 *
 * They describe the DATA, not a dialect, so they live apart from either table
 * definition: src/db/schema.ts (Postgres, what the Worker serves) and
 * src/db/schema.sqlite.ts (the offline pipeline's intermediate) both import
 * them, and neither has to depend on the other.
 */

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
