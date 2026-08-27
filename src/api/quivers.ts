/**
 * Quiver listing machinery + /quivers routes.
 *
 * The list envelope and item shapes mirror the legacy FastAPI backend
 * exactly (git history: qmd/schemas.py, qmd/crud.py): the contract is qmd_id,
 * num_vertices, exchange_matrix, class_size (null => rendered as ∞), ...
 * The same filter set serves /quivers, /search, and /export.
 */

import {
  and, asc, desc, eq, gt, inArray, lte, sql, type SQL,
} from "drizzle-orm";
import { Hono, type Context } from "hono";
import {
  mutationClasses as mc,
  mutationClassPayloads as payloads,
  quivers as q,
  type Matrix,
} from "../db/schema";
import { dbFor, dbForId, type Database } from "../db/shard";

// ---------------------------------------------------------------------------
// Query-param parsing (FastAPI-compatible: 'true'/'false', ints)
// ---------------------------------------------------------------------------

export class BadRequest extends Error {}

function parseBool(name: string, v: string | undefined): boolean | undefined {
  if (v === undefined || v === "") return undefined;
  const s = v.toLowerCase();
  if (s === "true" || s === "1") return true;
  if (s === "false" || s === "0") return false;
  throw new BadRequest(`${name} must be true or false`);
}

function parseInteger(name: string, v: string | undefined): number | undefined {
  if (v === undefined || v === "") return undefined;
  const i = Number(v);
  if (!Number.isInteger(i)) throw new BadRequest(`${name} must be an integer`);
  return i;
}

export interface ListFilters {
  rank?: number;
  dynkinType?: string;
  representationType?: string;
  maxEdge?: number;
  isOpen?: boolean;
  orbitMin?: number;
  orbitMax?: number;
  isAcyclic?: boolean;
  isConnected?: boolean;
  isSimplyLaced?: boolean;
  isMutationFinite?: boolean;
}

/** Parse the shared filter set from query params (union of /quivers + /search). */
export function parseFilters(get: (k: string) => string | undefined): ListFilters {
  return {
    rank: parseInteger("rank", get("rank")),
    dynkinType: get("dynkin_type") || undefined,
    representationType: get("representation_type") || undefined,
    maxEdge: parseInteger("max_edge", get("max_edge")),
    isOpen: parseBool("is_open", get("is_open")),
    orbitMin: parseInteger("orbit_min", get("orbit_min")),
    orbitMax: parseInteger("orbit_max", get("orbit_max")),
    isAcyclic: parseBool("is_acyclic", get("is_acyclic")),
    isConnected: parseBool("is_connected", get("is_connected")),
    isSimplyLaced: parseBool("is_simply_laced", get("is_simply_laced")),
    isMutationFinite: parseBool("is_mutation_finite", get("is_mutation_finite")),
  };
}

/** WHERE conditions matching the legacy backend's _filtered_quivers. */
export function filterConditions(f: ListFilters): SQL[] {
  const conds: SQL[] = [];
  if (f.rank !== undefined) conds.push(eq(q.n, f.rank));
  if (f.maxEdge !== undefined) conds.push(eq(q.maxEdge, f.maxEdge));
  if (f.isAcyclic !== undefined) conds.push(eq(q.isAcyclic, f.isAcyclic));
  if (f.isConnected !== undefined) conds.push(eq(q.isConnected, f.isConnected));
  if (f.isSimplyLaced !== undefined) {
    conds.push(f.isSimplyLaced ? lte(q.maxEdge, 1) : gt(q.maxEdge, 1));
  }
  // Class-side filters exclude quivers without a class, as in Postgres
  // (an inner comparison against a NULL outer-join row never matches).
  if (f.isOpen !== undefined) conds.push(eq(mc.isOpen, f.isOpen));
  // Mutation-finiteness filters on the *proved* columns, never on is_open:
  // a class that is only is_infinite_expected matches neither value.
  if (f.isMutationFinite !== undefined) {
    conds.push(f.isMutationFinite
      ? eq(mc.isFiniteConfirmed, true)
      : eq(mc.isInfiniteConfirmed, true));
  }
  if (f.dynkinType !== undefined) conds.push(eq(mc.dynkinType, f.dynkinType));
  if (f.representationType !== undefined) {
    conds.push(eq(q.representationType, f.representationType));
  }
  if (f.orbitMin !== undefined) conds.push(sql`${mc.classSize} >= ${f.orbitMin}`);
  if (f.orbitMax !== undefined) conds.push(sql`${mc.classSize} <= ${f.orbitMax}`);
  return conds;
}

// Frontend sort keys -> columns (legacy _SORT_COLUMNS).
const SORT_COLUMNS = {
  qmd_id: q.id,
  num_vertices: q.n,
  class_size: mc.classSize,
  max_edge: q.maxEdge,
  dynkin_type: mc.dynkinType,
  class_type: mc.isOpen,      // browse.html "Class" column (finite/open)
} as const;

/** Whitelisted sort column + direction; unknown values are a 400, not a silent fallback. */
export function sortOrder(sort: string | undefined, dir: string | undefined) {
  const key = sort ?? "num_vertices";
  if (!Object.hasOwn(SORT_COLUMNS, key)) {
    throw new BadRequest(`sort must be one of ${Object.keys(SORT_COLUMNS).join(", ")}`);
  }
  if (dir !== undefined && dir !== "asc" && dir !== "desc") {
    throw new BadRequest("dir must be 'asc' or 'desc'");
  }
  const col = SORT_COLUMNS[key as keyof typeof SORT_COLUMNS];
  return [dir === "desc" ? desc(col) : asc(col), asc(q.id)];
}

// ---------------------------------------------------------------------------
// Row selection + serializer
// ---------------------------------------------------------------------------

const LIST_SELECTION = {
  id: q.id,
  n: q.n,
  exchangeMatrix: q.exchangeMatrix,
  maxEdge: q.maxEdge,
  isAcyclic: q.isAcyclic,
  isConnected: q.isConnected,
  isBipartite: q.isBipartite,
  labelingCount: q.labelingCount,
  representationType: q.representationType,
  mcId: q.mutationClassId,
  mcIsOpen: mc.isOpen,
  mcDynkinType: mc.dynkinType,
  mcClassSize: mc.classSize,
};

type ListRow = {
  id: string; n: number; exchangeMatrix: Matrix; maxEdge: number;
  isAcyclic: boolean; isConnected: boolean; isBipartite: boolean | null;
  labelingCount: number | null; representationType: string | null;
  mcId: string | null; mcIsOpen: boolean | null; mcDynkinType: string | null;
  mcClassSize: number | null;
};

/** Labeled orbit size for closed classes; null (=> ∞) for open / unknown. */
function classSize(row: ListRow): number | null {
  if (row.mcIsOpen === null || row.mcIsOpen) return null;
  return row.mcClassSize;
}

export function quiverListItem(row: ListRow, matrix?: Matrix) {
  return {
    qmd_id: row.id,
    num_vertices: row.n,
    dynkin_type: row.mcDynkinType,
    representation_type: row.representationType,
    max_edge: row.maxEdge,
    is_acyclic: row.isAcyclic,
    is_connected: row.isConnected,
    is_bipartite: row.isBipartite,
    is_open: row.mcIsOpen ?? false,
    class_size: classSize(row),
    exchange_matrix: matrix ?? row.exchangeMatrix,
    mc_id: row.mcId,
  };
}

// ---------------------------------------------------------------------------
// Listing (shared by /quivers, /search, /export)
// ---------------------------------------------------------------------------

export interface ListParams {
  filters: ListFilters;
  scope: "distinct" | "labelings";
  sort?: string;
  dir?: string;
  offset: number;
  limit: number;
}

export async function listQuivers(db: Database, p: ListParams) {
  const conds = filterConditions(p.filters);
  const where = conds.length ? and(...conds) : undefined;
  const order = sortOrder(p.sort, p.dir);

  const totals = (await db
    .select({
      distinct: sql<number>`count(*)`,
      labeled: sql<number>`coalesce(sum(${q.labelingCount}), 0)`,
    })
    .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id))
    .where(where))[0] ?? { distinct: 0, labeled: 0 };

  if (p.scope === "labelings") {
    const items = await listLabelings(db, where, order, p.offset, p.limit);
    return { items, total: totals.labeled,
             distinct_total: totals.distinct, labeled_total: totals.labeled };
  }

  const rows = (await db
    .select(LIST_SELECTION)
    .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id))
    .where(where).orderBy(...order)
    .offset(p.offset).limit(p.limit)) as ListRow[];

  return {
    items: rows.map((r) => quiverListItem(r)),
    total: totals.distinct,
    distinct_total: totals.distinct,
    labeled_total: totals.labeled,
  };
}

/**
 * "labelings" scope: one row per labeled matrix in each quiver's class.
 *
 * labeling_count (stored per quiver) gives each quiver's expansion factor,
 * so the page window is located on the skinny rows first and only the
 * payloads (full orbits) of the classes actually on the page are loaded.
 * Loading all matching skinny rows is fine at current data size; a future
 * per-`n` cursor can replace the in-memory walk without changing the shape.
 */
async function listLabelings(
  db: Database, where: SQL | undefined, order: ReturnType<typeof sortOrder>,
  offset: number, limit: number,
) {
  const rows = (await db
    .select(LIST_SELECTION)
    .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id))
    .where(where).orderBy(...order)) as ListRow[];

  // Locate the quivers whose expansions intersect [offset, offset + limit).
  interface Window { row: ListRow; skip: number; take: number }
  const windows: Window[] = [];
  let pos = 0;
  for (const row of rows) {
    if (windows.length && pos >= offset + limit) break;
    const count = row.mcId ? Math.max(row.labelingCount ?? 1, 1) : 1;
    const start = pos;
    pos += count;
    if (pos <= offset) continue;
    const skip = Math.max(offset - start, 0);
    const take = Math.min(count - skip, offset + limit - Math.max(start, offset));
    if (take > 0) windows.push({ row, skip, take });
  }

  const mcIds = [...new Set(windows.map((w) => w.row.mcId)
    .filter((x): x is string => x !== null))];
  const orbitByMc = new Map<string, { qmd_id: string; matrix: Matrix }[]>();
  for (let i = 0; i < mcIds.length; i += 50) {
    const batch = await db
      .select({ id: payloads.mutationClassId, labeled: payloads.labeledQuivers })
      .from(payloads).where(inArray(payloads.mutationClassId, mcIds.slice(i, i + 50)));
    for (const b of batch) orbitByMc.set(b.id, b.labeled);
  }

  const items: ReturnType<typeof quiverListItem>[] = [];
  for (const { row, skip, take } of windows) {
    const labs = row.mcId
      ? (orbitByMc.get(row.mcId) ?? []).filter((e) => e.qmd_id === row.id)
      : [];
    if (labs.length === 0) {
      items.push(quiverListItem(row));
      continue;
    }
    for (const e of labs.slice(skip, skip + take)) {
      items.push(quiverListItem(row, e.matrix));
    }
  }
  return items;
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

export const quiversRoutes = new Hono<{ Bindings: Env }>();

/** Shared handler for GET /quivers and GET /search (different default limits). */
export function listHandler(defaultLimit: number) {
  return async (c: Context<{ Bindings: Env }>) => {
    const get = (k: string) => c.req.query(k);
    const scope = get("scope") ?? "distinct";
    if (scope !== "distinct" && scope !== "labelings") {
      return c.json({ detail: "scope must be 'distinct' or 'labelings'" }, 400);
    }
    const params: ListParams = {
      filters: parseFilters(get),
      scope,
      sort: get("sort"),
      dir: get("dir"),
      offset: Math.max(parseInteger("offset", get("offset")) ?? 0, 0),
      limit: Math.min(Math.max(parseInteger("limit", get("limit")) ?? defaultLimit, 1), 1000),
    };
    const db = dbFor(c.env, params.filters.rank ?? 0);
    return c.json(await listQuivers(db, params));
  };
}

quiversRoutes.get("/", listHandler(50));

quiversRoutes.get("/:id", async (c) => {
  const id = c.req.param("id");
  const db = dbForId(c.env, id);
  if (!db) return c.json({ detail: "Quiver not found" }, 404);

  const row = (await db
    .select({
      ...LIST_SELECTION,
      isAbundant: q.isAbundant,
      isPlanar: q.isPlanar,
      symmetryGroup: q.symmetryGroup,
      mcLabel: mc.label,
    })
    .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id))
    .where(eq(q.id, id)))[0];
  if (!row) return c.json({ detail: "Quiver not found" }, 404);

  // Shape: the legacy QuiverDetail response, field for field.
  return c.json({
    qmd_id: row.id,
    label: row.mcLabel,
    num_vertices: row.n,
    exchange_matrix: row.exchangeMatrix,
    dynkin_type: row.mcDynkinType,
    is_open: row.mcIsOpen ?? false,
    is_acyclic: row.isAcyclic,
    is_connected: row.isConnected,
    max_edge: row.maxEdge,
    is_bipartite: row.isBipartite,
    is_abundant: row.isAbundant,
    is_planar: row.isPlanar,
    representation_type: row.representationType,
    symmetry_group: row.symmetryGroup,
    class_size: classSize(row as ListRow),
    mc_id: row.mcId,
    tags: [],
  });
});
