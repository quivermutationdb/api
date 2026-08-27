/**
 * Quiver listing machinery + /quivers routes.
 *
 * The list envelope and item shapes mirror the legacy FastAPI backend
 * (git history: qmd/schemas.py, qmd/crud.py): qmd_id, num_vertices,
 * exchange_matrix, class_size (null => rendered as ∞), ... — with additive
 * fields (`nickname`, `exploration`, `next_cursor`) for phase 2.
 * The same filter set serves /quivers, /search, /export and the MCP tools.
 */

import { and, eq, gt, lte, sql, type SQL } from "drizzle-orm";
import { Hono, type Context } from "hono";
import {
  classNicknames as nick,
  labelings as lab,
  mutationClasses as mc,
  quivers as q,
  rankStats,
  type Matrix,
} from "../db/schema";
import { dbFor, dbForId, type Database } from "../db/shard";
import { afterKey, decodeCursor, encodeCursor, orderBy } from "./cursor";
import { BadRequest, parseBool, parseDir, parseInteger, parsePaging } from "./errors";

export { BadRequest } from "./errors";

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

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
  nickname?: string;
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
    nickname: get("nickname") || undefined,
  };
}

/** The applied cut as it is logged / echoed (non-undefined filters only). */
export function filtersAsRecord(f: ListFilters): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries({
    rank: f.rank, dynkin_type: f.dynkinType,
    representation_type: f.representationType, max_edge: f.maxEdge,
    is_open: f.isOpen, orbit_min: f.orbitMin, orbit_max: f.orbitMax,
    is_acyclic: f.isAcyclic, is_connected: f.isConnected,
    is_simply_laced: f.isSimplyLaced, is_mutation_finite: f.isMutationFinite,
    nickname: f.nickname,
  })) {
    if (v !== undefined) out[k] = v;
  }
  return out;
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
  // Class-side filters exclude quivers without a class (a comparison against
  // a NULL outer-join row never matches).
  if (f.isOpen !== undefined) conds.push(eq(mc.isOpen, f.isOpen));
  // Mutation-finiteness filters on the *proved* columns, never on is_open:
  // a class that is only is_infinite_expected (or truncated) matches neither.
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
  if (f.nickname !== undefined) conds.push(eq(nick.slug, f.nickname.toLowerCase()));
  return conds;
}

/** True when the cut is "everything" or "one rank" — totals then come from rank_stats. */
function onlyRankFilter(f: ListFilters): boolean {
  return Object.keys(filtersAsRecord(f)).every((k) => k === "rank");
}

// ---------------------------------------------------------------------------
// Sorting (whitelisted) + keyset keys
// ---------------------------------------------------------------------------

const SORT_COLUMNS = {
  qmd_id: q.id,
  num_vertices: q.n,
  class_size: mc.classSize,
  max_edge: q.maxEdge,
  dynkin_type: mc.dynkinType,
  class_type: mc.isOpen,      // browse.html "Class" column (finite/open)
} as const;
export type SortKey = keyof typeof SORT_COLUMNS;

export function parseSort(sort: string | undefined): SortKey {
  const key = sort ?? "num_vertices";
  if (!Object.hasOwn(SORT_COLUMNS, key)) {
    throw new BadRequest(`sort must be one of ${Object.keys(SORT_COLUMNS).join(", ")}`);
  }
  return key as SortKey;
}

/** ORDER BY columns for a sort: the sort column, then (n, id) as a unique tiebreak. */
function sortColumns(key: SortKey, dir: "asc" | "desc") {
  if (key === "num_vertices") return { cols: [q.n, q.id], dirs: [dir, "asc" as const] };
  if (key === "qmd_id") return { cols: [q.id], dirs: [dir] };
  return { cols: [SORT_COLUMNS[key], q.n, q.id], dirs: [dir, "asc" as const, "asc" as const] };
}

/** ORDER BY for a sort + direction (used by the export). */
export function sortOrder(sort: string | undefined, dir: string | undefined) {
  const s = sortColumns(parseSort(sort), parseDir(dir));
  return orderBy(s.cols, s.dirs);
}

// ---------------------------------------------------------------------------
// Row selection + serializer
// ---------------------------------------------------------------------------

export const LIST_SELECTION = {
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
  mcExploration: mc.exploration,
  mcDynkinType: mc.dynkinType,
  mcClassSize: mc.classSize,
  nickname: nick.nickname,
  nicknameSlug: nick.slug,
};

export type ListRow = {
  id: string; n: number; exchangeMatrix: Matrix; maxEdge: number;
  isAcyclic: boolean; isConnected: boolean; isBipartite: boolean | null;
  labelingCount: number | null; representationType: string | null;
  mcId: string | null; mcIsOpen: boolean | null; mcExploration: string | null;
  mcDynkinType: string | null; mcClassSize: number | null;
  nickname: string | null; nicknameSlug: string | null;
};

/** Labeled orbit size for completely explored classes; null (=> ∞ / unknown) otherwise. */
export function classSize(row: { mcIsOpen: boolean | null; mcClassSize: number | null }): number | null {
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
    exploration: row.mcExploration,
    class_size: classSize(row),
    explored_size: row.mcClassSize,
    exchange_matrix: matrix ?? row.exchangeMatrix,
    mc_id: row.mcId,
    nickname: row.nickname,
    nickname_slug: row.nicknameSlug,
  };
}

function baseQuery(db: Database) {
  return db.select(LIST_SELECTION).from(q)
    .leftJoin(mc, eq(q.mutationClassId, mc.id))
    .leftJoin(nick, eq(nick.mcId, mc.id));
}

// ---------------------------------------------------------------------------
// Listing (shared by /quivers, /search, MCP)
// ---------------------------------------------------------------------------

export interface ListParams {
  filters: ListFilters;
  scope: "distinct" | "labelings";
  sort?: string;
  dir?: string;
  offset: number;
  limit: number;
  cursor?: string;
}

async function totalsFor(db: Database, f: ListFilters, where: SQL | undefined) {
  if (onlyRankFilter(f)) {
    const rows = await db.select().from(rankStats)
      .where(f.rank !== undefined ? eq(rankStats.n, f.rank) : undefined);
    return {
      distinct: rows.reduce((a, r) => a + r.quiverCount, 0),
      labeled: rows.reduce((a, r) => a + r.labeledQuiverCount, 0),
    };
  }
  return (await db
    .select({
      distinct: sql<number>`count(*)`,
      labeled: sql<number>`coalesce(sum(${q.labelingCount}), 0)`,
    })
    .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id))
    .leftJoin(nick, eq(nick.mcId, mc.id))
    .where(where))[0] ?? { distinct: 0, labeled: 0 };
}

function keyValue(row: ListRow, col: unknown): string | number | null {
  switch (col) {
    case q.id: return row.id;
    case q.n: return row.n;
    case q.maxEdge: return row.maxEdge;
    case mc.classSize: return row.mcClassSize;
    case mc.dynkinType: return row.mcDynkinType;
    case mc.isOpen: return row.mcIsOpen === null ? null : Number(row.mcIsOpen);
    default: throw new Error("unknown key column");
  }
}

export async function listQuivers(db: Database, p: ListParams) {
  const conds = filterConditions(p.filters);
  const sortKey = parseSort(p.sort);
  const dir = parseDir(p.dir);
  const where = conds.length ? and(...conds) : undefined;
  const totals = await totalsFor(db, p.filters, where);

  if (p.scope === "labelings") {
    if (sortKey !== "num_vertices" || dir !== "asc") {
      throw new BadRequest("scope=labelings supports only the default sort (num_vertices asc)");
    }
    const r = await listLabelings(db, conds, p);
    return { items: r.items, total: totals.labeled, distinct_total: totals.distinct,
             labeled_total: totals.labeled, next_cursor: r.next_cursor };
  }

  const { cols, dirs } = sortColumns(sortKey, dir);
  const after = decodeCursor(p.cursor, cols.length);
  const fullWhere = after ? and(where, afterKey(cols, dirs, after)) : where;
  const rows = (await baseQuery(db)
    .where(fullWhere).orderBy(...orderBy(cols, dirs))
    .offset(after ? 0 : p.offset).limit(p.limit + 1)) as ListRow[];

  const page = rows.slice(0, p.limit);
  const last = page[page.length - 1];
  const next_cursor = rows.length > p.limit && last
    ? encodeCursor(cols.map((c) => keyValue(last, c)))
    : null;
  return {
    items: page.map((r) => quiverListItem(r)),
    total: totals.distinct,
    distinct_total: totals.distinct,
    labeled_total: totals.labeled,
    next_cursor,
  };
}

/**
 * "labelings" scope: one row per labeled matrix, straight from the labelings
 * table in (n, id, ord) order — served by idx_q_n_id + idx_lab_qmd_ord, no
 * in-memory walk. Keyset key: [n, qmd_id, ord].
 */
async function listLabelings(db: Database, conds: SQL[], p: ListParams) {
  const cols = [q.n, q.id, lab.ord];
  const dirs: ("asc" | "desc")[] = ["asc", "asc", "asc"];
  const after = decodeCursor(p.cursor, 3);
  const where = and(...conds, after ? afterKey(cols, dirs, after) : undefined);
  const rows = await db
    .select({ ...LIST_SELECTION, ord: lab.ord, labMatrix: lab.matrix })
    .from(lab)
    .innerJoin(q, eq(q.id, lab.qmdId))
    .leftJoin(mc, eq(q.mutationClassId, mc.id))
    .leftJoin(nick, eq(nick.mcId, mc.id))
    .where(where).orderBy(...orderBy(cols, dirs))
    .offset(after ? 0 : p.offset).limit(p.limit + 1);
  const page = rows.slice(0, p.limit);
  const last = page[page.length - 1];
  return {
    items: page.map((r) => ({ ...quiverListItem(r as ListRow, r.labMatrix), labeling_ord: r.ord })),
    next_cursor: rows.length > p.limit && last ? encodeCursor([last.n, last.id, last.ord]) : null,
  };
}

// ---------------------------------------------------------------------------
// Labelings of one quiver (paged)
// ---------------------------------------------------------------------------

export async function quiverLabelings(db: Database, qmdId: string, cursor: string | undefined, limit: number) {
  const after = decodeCursor(cursor, 1);
  const rows = await db.select({ mcId: lab.mutationClassId, ord: lab.ord, matrix: lab.matrix })
    .from(lab)
    .where(and(eq(lab.qmdId, qmdId), after ? gt(lab.ord, after[0] as number) : undefined))
    .orderBy(lab.ord).limit(limit + 1);
  const page = rows.slice(0, limit);
  const last = page[page.length - 1];
  return {
    items: page.map((r) => ({ mc_id: r.mcId, ord: r.ord, matrix: r.matrix })),
    next_cursor: rows.length > limit && last ? encodeCursor([last.ord]) : null,
  };
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

export const quiversRoutes = new Hono<{ Bindings: Env }>();

export function listParamsFrom(get: (k: string) => string | undefined, defaultLimit: number): ListParams {
  const scope = get("scope") ?? "distinct";
  if (scope !== "distinct" && scope !== "labelings") {
    throw new BadRequest("scope must be 'distinct' or 'labelings'");
  }
  return {
    filters: parseFilters(get),
    scope,
    sort: get("sort"),
    dir: get("dir"),
    cursor: get("cursor"),
    ...parsePaging(get, defaultLimit),
  };
}

/** Shared handler for GET /quivers and GET /search (different default limits). */
export function listHandler(defaultLimit: number) {
  return async (c: Context<{ Bindings: Env }>) => {
    const params = listParamsFrom((k) => c.req.query(k), defaultLimit);
    const db = dbFor(c.env, params.filters.rank ?? 0);
    return c.json(await listQuivers(db, params));
  };
}

quiversRoutes.get("/", listHandler(50));

export async function quiverDetail(db: Database, id: string) {
  const row = (await db
    .select({
      ...LIST_SELECTION,
      isAbundant: q.isAbundant,
      isPlanar: q.isPlanar,
      symmetryGroup: q.symmetryGroup,
      mcLabel: mc.label,
    })
    .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id))
    .leftJoin(nick, eq(nick.mcId, mc.id))
    .where(eq(q.id, id)))[0];
  if (!row) return null;
  // Shape: the legacy QuiverDetail response, field for field (+ additive fields).
  return {
    qmd_id: row.id,
    label: row.mcLabel,
    num_vertices: row.n,
    exchange_matrix: row.exchangeMatrix,
    dynkin_type: row.mcDynkinType,
    is_open: row.mcIsOpen ?? false,
    exploration: row.mcExploration,
    is_acyclic: row.isAcyclic,
    is_connected: row.isConnected,
    max_edge: row.maxEdge,
    is_bipartite: row.isBipartite,
    is_abundant: row.isAbundant,
    is_planar: row.isPlanar,
    representation_type: row.representationType,
    symmetry_group: row.symmetryGroup,
    class_size: classSize(row),
    explored_size: row.mcClassSize,
    labeling_count: row.labelingCount,
    mc_id: row.mcId,
    nickname: row.nickname,
    nickname_slug: row.nicknameSlug,
    tags: [] as string[],
  };
}

quiversRoutes.get("/:id", async (c) => {
  const id = c.req.param("id");
  const db = dbForId(c.env, id);
  const detail = db ? await quiverDetail(db, id) : null;
  if (!detail) return c.json({ detail: "Quiver not found" }, 404);
  return c.json(detail);
});

quiversRoutes.get("/:id/labelings", async (c) => {
  const id = c.req.param("id");
  const db = dbForId(c.env, id);
  if (!db) return c.json({ detail: "Quiver not found" }, 404);
  const exists = (await db.select({ id: q.id }).from(q).where(eq(q.id, id)))[0];
  if (!exists) return c.json({ detail: "Quiver not found" }, 404);
  const { limit } = parsePaging((k) => c.req.query(k), 100);
  return c.json({ qmd_id: id, ...(await quiverLabelings(db, id, c.req.query("cursor"), limit)) });
});
