/**
 * Quiver listing machinery + /quivers routes (schema v3, sharded).
 *
 * The list envelope and item shapes mirror the legacy FastAPI backend
 * (qmd_id, num_vertices, exchange_matrix, class_size (null => ∞), ...) with
 * additive phase-2/3 fields (nickname, exploration, mutation_finite,
 * next_cursor). Matrices are stored compactly and decoded here. Lists run
 * per shard (src/db/shard.ts) and are merged by sort key (src/api/merge.ts);
 * the rowid is the unique tiebreak (rows are inserted in id order per rank).
 */

import { and, eq, gt, lte, sql, type SQL } from "drizzle-orm";
import { Hono, type Context } from "hono";
import { decodeUpper } from "../db/matrix";
import {
  classNicknames as nick,
  labelings as lab,
  mutationClasses as mc,
  quivers as q,
  rankStats,
  type Matrix,
} from "../db/schema";
import { dbForId, dbOf, mainDb, shardsForRank, type Database, type Shard } from "../db/shard";
import { afterKey, decodeCursor, encodeCursor, orderBy, type Dir, type Key, type KeyCol } from "./cursor";
import { BadRequest, parseBool, parseDir, parseInteger, parsePaging } from "./errors";
import { mergeShards } from "./merge";

export { BadRequest } from "./errors";

const Q_ROWID = sql`${q}.rowid`;

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
  explored?: boolean;
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
    explored: parseBool("explored", get("explored")),
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
    nickname: f.nickname, explored: f.explored,
  })) {
    if (v !== undefined) out[k] = v;
  }
  return out;
}

/** WHERE conditions (the legacy backend's _filtered_quivers, extended). */
export function filterConditions(f: ListFilters): SQL[] {
  const conds: SQL[] = [];
  if (f.rank !== undefined) conds.push(eq(q.n, f.rank));
  if (f.maxEdge !== undefined) conds.push(eq(q.maxEdge, f.maxEdge));
  if (f.isAcyclic !== undefined) conds.push(eq(q.isAcyclic, f.isAcyclic));
  if (f.isConnected !== undefined) conds.push(eq(q.isConnected, f.isConnected));
  if (f.isSimplyLaced !== undefined) conds.push(f.isSimplyLaced ? lte(q.maxEdge, 1) : gt(q.maxEdge, 1));
  // Per-quiver finiteness is known even without a class row (Derksen–Owen).
  if (f.isMutationFinite !== undefined) conds.push(eq(q.mutationFinite, f.isMutationFinite));
  if (f.explored !== undefined) {
    conds.push(f.explored ? sql`${q.mutationClassId} is not null` : sql`${q.mutationClassId} is null`);
  }
  // Class-side filters exclude quivers without an explored class.
  if (f.isOpen !== undefined) conds.push(eq(mc.isOpen, f.isOpen));
  if (f.dynkinType !== undefined) conds.push(eq(mc.dynkinType, f.dynkinType));
  if (f.representationType !== undefined) conds.push(eq(q.representationType, f.representationType));
  if (f.orbitMin !== undefined) conds.push(sql`${mc.classSize} >= ${f.orbitMin}`);
  if (f.orbitMax !== undefined) conds.push(sql`${mc.classSize} <= ${f.orbitMax}`);
  if (f.nickname !== undefined) conds.push(eq(nick.slug, f.nickname.toLowerCase()));
  return conds;
}

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

/** ORDER BY columns for a sort: the sort column, then (n, rowid) as the unique tiebreak. */
function sortColumns(key: SortKey, dir: Dir): { cols: KeyCol[]; dirs: Dir[] } {
  if (key === "num_vertices") return { cols: [q.n, Q_ROWID], dirs: [dir, "asc"] };
  if (key === "qmd_id") return { cols: [q.id], dirs: [dir] };
  return { cols: [SORT_COLUMNS[key], q.n, Q_ROWID], dirs: [dir, "asc", "asc"] };
}

// ---------------------------------------------------------------------------
// Row selection + serializer
// ---------------------------------------------------------------------------

export const LIST_SELECTION = {
  rowid: sql<number>`${q}.rowid`,
  id: q.id,
  n: q.n,
  exchangeMatrix: q.exchangeMatrix,
  maxEdge: q.maxEdge,
  isAcyclic: q.isAcyclic,
  isConnected: q.isConnected,
  isBipartite: q.isBipartite,
  labelingCount: q.labelingCount,
  mutationFinite: q.mutationFinite,
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
  rowid: number; id: string; n: number; exchangeMatrix: string; maxEdge: number;
  isAcyclic: boolean; isConnected: boolean; isBipartite: boolean | null;
  labelingCount: number | null; mutationFinite: boolean | null; representationType: string | null;
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
    exploration: row.mcId ? row.mcExploration : null,
    explored: row.mcId !== null,
    mutation_finite: row.mutationFinite,
    class_size: classSize(row),
    explored_size: row.mcClassSize,
    exchange_matrix: matrix ?? decodeUpper(row.n, row.exchangeMatrix),
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

function keyValue(row: ListRow, col: unknown): string | number | null {
  switch (col) {
    case q.id: return row.id;
    case q.n: return row.n;
    case q.maxEdge: return row.maxEdge;
    case mc.classSize: return row.mcClassSize;
    case mc.dynkinType: return row.mcDynkinType;
    case mc.isOpen: return row.mcIsOpen === null ? null : Number(row.mcIsOpen);
    case Q_ROWID: return row.rowid;
    default: throw new Error("unknown key column");
  }
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

async function totalsFor(env: Env, f: ListFilters, where: SQL | undefined) {
  if (onlyRankFilter(f)) {
    const rows = await mainDb(env).select().from(rankStats)
      .where(f.rank !== undefined ? eq(rankStats.n, f.rank) : undefined);
    return {
      distinct: rows.reduce((a, r) => a + r.quiverCount, 0),
      labeled: rows.reduce((a, r) => a + r.labeledQuiverCount, 0),
    };
  }
  const per = await Promise.all(shardsForRank(f.rank).map((s) => dbOf(env, s)
    .select({ distinct: sql<number>`count(*)`, labeled: sql<number>`coalesce(sum(${q.labelingCount}), 0)` })
    .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id)).leftJoin(nick, eq(nick.mcId, mc.id))
    .where(where)));
  return {
    distinct: per.reduce((a, r) => a + (r[0]?.distinct ?? 0), 0),
    labeled: per.reduce((a, r) => a + (r[0]?.labeled ?? 0), 0),
  };
}

export async function listQuivers(env: Env, p: ListParams) {
  const conds = filterConditions(p.filters);
  const sortKey = parseSort(p.sort);
  const dir = parseDir(p.dir);
  const where = conds.length ? and(...conds) : undefined;
  const totals = await totalsFor(env, p.filters, where);
  const shards = shardsForRank(p.filters.rank);

  if (p.scope === "labelings") {
    if (sortKey !== "num_vertices" || dir !== "asc") {
      throw new BadRequest("scope=labelings supports only the default sort (num_vertices asc)");
    }
    const r = await listLabelings(env, shards, conds, p);
    return { items: r.items, total: totals.labeled, distinct_total: totals.distinct,
             labeled_total: totals.labeled, next_cursor: r.next_cursor };
  }

  const { cols, dirs } = sortColumns(sortKey, dir);
  const r = await mergeShards<ListRow>({
    shardKeys: shards.map((s) => s.key),
    dirs,
    keyOf: (row) => cols.map((c) => keyValue(row, c)),
    fetch: async (sk, after, limit) => (await baseQuery(dbOf(env, shards.find((s) => s.key === sk)!))
      .where(after ? and(where, afterKey(cols, dirs, after)) : where)
      .orderBy(...orderBy(cols, dirs)).limit(limit)) as ListRow[],
    limit: p.limit, offset: p.offset, cursor: p.cursor,
  });
  return {
    items: r.items.map((row) => quiverListItem(row)),
    total: totals.distinct,
    distinct_total: totals.distinct,
    labeled_total: totals.labeled,
    next_cursor: r.next_cursor,
  };
}

/**
 * "labelings" scope: one row per labeled matrix (complete classes only),
 * from the labelings table in (n, quiver rowid, ord) order. Key: [n, rowid, ord].
 */
async function listLabelings(env: Env, shards: Shard[], conds: SQL[], p: ListParams) {
  const cols: KeyCol[] = [q.n, Q_ROWID, lab.ord];
  const dirs: Dir[] = ["asc", "asc", "asc"];
  type Row = ListRow & { ord: number; labMatrix: string };
  const r = await mergeShards<Row>({
    shardKeys: shards.map((s) => s.key),
    dirs,
    keyOf: (row) => [row.n, row.rowid, row.ord],
    fetch: async (sk, after, limit) => (await dbOf(env, shards.find((s) => s.key === sk)!)
      .select({ ...LIST_SELECTION, ord: lab.ord, labMatrix: lab.matrix })
      .from(lab).innerJoin(q, eq(q.id, lab.qmdId))
      .leftJoin(mc, eq(q.mutationClassId, mc.id)).leftJoin(nick, eq(nick.mcId, mc.id))
      .where(and(...conds, after ? afterKey(cols, dirs, after) : undefined))
      .orderBy(...orderBy(cols, dirs)).limit(limit)) as Row[],
    limit: p.limit, offset: p.offset, cursor: p.cursor,
  });
  return {
    items: r.items.map((row) => ({ ...quiverListItem(row, decodeUpper(row.n, row.labMatrix)), labeling_ord: row.ord })),
    next_cursor: r.next_cursor,
  };
}

// ---------------------------------------------------------------------------
// Labelings of one quiver (paged; a quiver's labelings live in its class's shard)
// ---------------------------------------------------------------------------

export async function quiverLabelings(env: Env, qmdId: string, mcId: string | null,
                                      cursor: string | undefined, limit: number) {
  const db = mcId ? dbForId(env, mcId) : null;
  if (!db) return { items: [], next_cursor: null };
  const after = decodeCursor(cursor, 1);
  const n = Number(/^Q\.n(\d+)\./.exec(qmdId)?.[1] ?? 0);
  const rows = await db.select({ mcId: lab.mutationClassId, ord: lab.ord, matrix: lab.matrix })
    .from(lab)
    .where(and(eq(lab.qmdId, qmdId), after ? gt(lab.ord, after[0] as number) : undefined))
    .orderBy(lab.ord).limit(limit + 1);
  const page = rows.slice(0, limit);
  const last = page[page.length - 1];
  return {
    items: page.map((r) => ({ mc_id: r.mcId, ord: r.ord, matrix: decodeUpper(n, r.matrix) })),
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
    filters: parseFilters(get), scope, sort: get("sort"), dir: get("dir"), cursor: get("cursor"),
    ...parsePaging(get, defaultLimit),
  };
}

/** Shared handler for GET /quivers and GET /search (different default limits). */
export function listHandler(defaultLimit: number) {
  return async (c: Context<{ Bindings: Env }>) => {
    const params = listParamsFrom((k) => c.req.query(k), defaultLimit);
    return c.json(await listQuivers(c.env, params));
  };
}

quiversRoutes.get("/", listHandler(50));

export async function quiverDetail(env: Env, id: string) {
  const db = dbForId(env, id);
  if (!db) return null;
  const row = (await db
    .select({ ...LIST_SELECTION, isAbundant: q.isAbundant, isPlanar: q.isPlanar,
              symmetryGroup: q.symmetryGroup, mcLabel: mc.label })
    .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id)).leftJoin(nick, eq(nick.mcId, mc.id))
    .where(eq(q.id, id)))[0];
  if (!row) return null;
  // Shape: the legacy QuiverDetail response, field for field (+ additive fields).
  return {
    qmd_id: row.id,
    label: row.mcLabel,
    num_vertices: row.n,
    exchange_matrix: decodeUpper(row.n, row.exchangeMatrix),
    dynkin_type: row.mcDynkinType,
    is_open: row.mcIsOpen ?? false,
    exploration: row.mcId ? row.mcExploration : null,
    explored: row.mcId !== null,
    mutation_finite: row.mutationFinite,
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
  const detail = await quiverDetail(c.env, c.req.param("id"));
  if (!detail) return c.json({ detail: "Quiver not found" }, 404);
  return c.json(detail);
});

quiversRoutes.get("/:id/labelings", async (c) => {
  const id = c.req.param("id");
  const db = dbForId(c.env, id);
  if (!db) return c.json({ detail: "Quiver not found" }, 404);
  const row = (await db.select({ id: q.id, mcId: q.mutationClassId }).from(q).where(eq(q.id, id)))[0];
  if (!row) return c.json({ detail: "Quiver not found" }, 404);
  const { limit } = parsePaging((k) => c.req.query(k), 100);
  return c.json({ qmd_id: id, ...(await quiverLabelings(c.env, id, row.mcId, c.req.query("cursor"), limit)) });
});

export type { Key };
