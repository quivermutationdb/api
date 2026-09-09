/**
 * Quiver listing machinery + /quivers routes (schema v3, sharded).
 *
 * The list envelope and item shapes mirror the legacy FastAPI backend
 * (qmd_id, num_vertices, exchange_matrix, class_size (null => ∞), ...) with
 * additive phase-2/3 fields (nickname, exploration, mutation_finite,
 * next_cursor). Matrices are stored compactly and decoded here. Lists are a
 * single keyset-paged query against the one Postgres database; `seq` is the
 * unique tiebreak, assigned at load so that (n, seq) is id order. Both the
 * per-shard fan-out and the JS merge that went with it died at the cutover.
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
import { dbForId, mainDb, type Database } from "../db/shard";
import { afterKey, decodeCursor, encodeCursor, keysetPage, orderBy,
         type Dir, type Key, type KeyCol } from "./cursor";
import { BadRequest, parseBool, parseDir, parseInteger, parsePaging, parseTotalMode,
         type TotalMode } from "./errors";

export { BadRequest } from "./errors";

const Q_SEQ = q.seq;

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
  if (f.rank !== undefined) {
    conds.push(eq(q.n, f.rank));
    // Redundant with n = rank, and deliberately so. Ids are `Q.n{k}.{16 hex}`
    // (an enforced invariant -- see CLAUDE.md), so every rank-k row lies in the
    // byte range ['Q.nk.', 'Q.nk/') under the C collation. Stating it gives the
    // planner a range it can seek on quivers_pkey; without it, ORDER BY id with
    // a rank filter scans the pkey from the start, past every lower-rank id,
    // because ids sort by rank. Measured on rank 6, where that means walking
    // 5.9 M rows first: **7,340 ms with the range removed, 1.6 ms with it.**
    conds.push(sql`${q.id} >= ${`Q.n${f.rank}.`} and ${q.id} < ${`Q.n${f.rank}/`}`);
  }
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

/** ORDER BY columns for a sort: the sort column, then (n, seq) as the unique tiebreak. */
function sortColumns(key: SortKey, dir: Dir): { cols: KeyCol[]; dirs: Dir[] } {
  if (key === "num_vertices") return { cols: [q.n, Q_SEQ], dirs: [dir, "asc"] };
  if (key === "qmd_id") return { cols: [q.id], dirs: [dir] };
  return { cols: [SORT_COLUMNS[key], q.n, Q_SEQ], dirs: [dir, "asc", "asc"] };
}

// ---------------------------------------------------------------------------
// Row selection + serializer
// ---------------------------------------------------------------------------

export const LIST_SELECTION = {
  seq: q.seq,
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
  seq: number | null; id: string; n: number; exchangeMatrix: string; maxEdge: number;
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
    case Q_SEQ: return row.seq;
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
  total: TotalMode;
}

/**
 * Result counts for a list response. Three paths, cheapest first:
 *
 *   - rank-only filter -> the ingest-time aggregates in `rank_stats`, no scan;
 *   - `?total=capped` (the default) -> count inside a `LIMIT TOTAL_CAP + 1`
 *     subquery, so the engine stops once the answer stops being interesting;
 *   - `?total=exact` -> a real count(*) over the filter.
 *
 * The cap exists because the exact count is a full scan of the quivers heap
 * whenever the filter is neither rank-only nor index-backed, and at rank 6
 * that heap is 9 GB. Measured single-threaded on the Phase 0 Postgres load,
 * `n = 6 AND is_acyclic` took 6.0 s exact against 12 ms capped at 10k. On a
 * PS-40 (0.5 vCPU, and 4 GB of RAM the heap does not fit in) the exact form is
 * tens of seconds: a browse page that never paints. docs/PLANETSCALE.md has
 * the measurements and the partial indexes that cover the other direction.
 *
 * `capped: true` means every number here is a LOWER BOUND, surfaced to clients
 * as `total_is_lower_bound`. They stay real counts of rows actually visited
 * rather than the cap itself, which is what keeps them meaningful under
 * sharding, where each shard is capped independently and the bounds add.
 */
export const TOTAL_CAP = 10_000;

interface Totals { distinct: number; labeled: number; capped: boolean }

async function totalsFor(env: Env, f: ListFilters, where: SQL | undefined,
                         mode: TotalMode): Promise<Totals> {
  if (onlyRankFilter(f)) {
    const rows = await mainDb(env).select().from(rankStats)
      .where(f.rank !== undefined ? eq(rankStats.n, f.rank) : undefined);
    // Number(): Postgres returns bigint (and any widened counter) as a STRING to
    // avoid precision loss, so `a + r.quiverCount` would concatenate. The
    // sql<number> annotations here are compile-time only -- TypeScript cannot
    // catch this, and the result is a plausible-looking wrong total.
    return {
      distinct: rows.reduce((a, r) => a + Number(r.quiverCount), 0),
      labeled: rows.reduce((a, r) => a + Number(r.labeledQuiverCount), 0),
      capped: false,
    };
  }
  return countIn(mainDb(env), where, mode);
}

async function countIn(db: Database, where: SQL | undefined, mode: TotalMode): Promise<Totals> {
  if (mode === "exact") {
    const r = (await db
      .select({ distinct: sql<number>`count(*)`,
                labeled: sql<number>`coalesce(sum(${q.labelingCount}), 0)` })
      .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id)).leftJoin(nick, eq(nick.mcId, mc.id))
      .where(where))[0];
    return { distinct: Number(r?.distinct ?? 0), labeled: Number(r?.labeled ?? 0), capped: false };
  }
  // The one row past the cap is what distinguishes "exactly TOTAL_CAP" from
  // "at least TOTAL_CAP"; it is dropped from the reported figure below.
  const sub = db.select({ lc: q.labelingCount }).from(q)
    .leftJoin(mc, eq(q.mutationClassId, mc.id)).leftJoin(nick, eq(nick.mcId, mc.id))
    .where(where).limit(TOTAL_CAP + 1).as("capped_rows");
  const r = (await db.select({
    distinct: sql<number>`count(*)`,
    labeled: sql<number>`coalesce(sum(${sub.lc}), 0)`,
  }).from(sub))[0];
  const seen = Number(r?.distinct ?? 0);
  const capped = seen > TOTAL_CAP;
  return { distinct: capped ? TOTAL_CAP : seen, labeled: Number(r?.labeled ?? 0), capped };
}

export async function listQuivers(env: Env, p: ListParams) {
  const conds = filterConditions(p.filters);
  const sortKey = parseSort(p.sort);
  const dir = parseDir(p.dir);
  const where = conds.length ? and(...conds) : undefined;

  // Validate before starting any query, so no promise is left dangling on throw.
  if (p.scope === "labelings" && (sortKey !== "num_vertices" || dir !== "asc")) {
    throw new BadRequest("scope=labelings supports only the default sort (num_vertices asc)");
  }

  // Started here, awaited with the page read below. Whenever the filter is not
  // rank-only, totalsFor falls back to a real count(*) over two left joins;
  // awaiting it up front put that scan on the critical path of every list
  // response instead of running it alongside the page.
  const totalsP = totalsFor(env, p.filters, where, p.total);

  if (p.scope === "labelings") {
    const [totals, r] = await Promise.all([totalsP, listLabelings(env, conds, p)]);
    return { items: r.items, total: totals.labeled, distinct_total: totals.distinct,
             labeled_total: totals.labeled, total_is_lower_bound: totals.capped,
             next_cursor: r.next_cursor };
  }

  const { cols, dirs } = sortColumns(sortKey, dir);
  const [totals, r] = await Promise.all([totalsP, keysetPage<ListRow>({
    dirs,
    keyOf: (row) => cols.map((c) => keyValue(row, c)),
    fetch: async (after, limit) => (await baseQuery(mainDb(env))
      .where(after ? and(where, afterKey(cols, dirs, after)) : where)
      .orderBy(...orderBy(cols, dirs)).limit(limit)) as ListRow[],
    limit: p.limit, offset: p.offset, cursor: p.cursor,
  })]);
  return {
    items: r.items.map((row) => quiverListItem(row)),
    total: totals.distinct,
    distinct_total: totals.distinct,
    labeled_total: totals.labeled,
    total_is_lower_bound: totals.capped,
    next_cursor: r.next_cursor,
  };
}

/**
 * "labelings" scope: one row per labeled matrix (complete classes only),
 * from the labelings table in (n, quiver seq, ord) order. Key: [n, seq, ord].
 */
async function listLabelings(env: Env, conds: SQL[], p: ListParams) {
  const cols: KeyCol[] = [q.n, Q_SEQ, lab.ord];
  const dirs: Dir[] = ["asc", "asc", "asc"];
  type Row = ListRow & { ord: number; labMatrix: string };
  const r = await keysetPage<Row>({
    dirs,
    keyOf: (row) => [row.n, row.seq, row.ord],
    fetch: async (after, limit) => (await mainDb(env)
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
    total: parseTotalMode(get("total")),
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
