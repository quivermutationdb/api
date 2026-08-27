/**
 * /classes routes (schema v3, sharded): browse list, detail, paged members,
 * nickname lookup. A class row and its labelings live in the shard of the
 * class id; its member quivers may live in any shard of the rank, so member
 * listings query every shard of the rank and merge.
 */

import { and, eq, gt, ne, sql, type SQL } from "drizzle-orm";
import { Hono, type Context } from "hono";
import { decodeUpper } from "../db/matrix";
import {
  classNicknames as nick,
  labelings as lab,
  mutationClasses as mc,
  quivers as q,
  rankStats,
} from "../db/schema";
import { dbForId, dbOf, mainDb, rankFromId, shardsForRank, type Database } from "../db/shard";
import { afterKey, decodeCursor, encodeCursor, orderBy, type Dir, type Key, type KeyCol } from "./cursor";
import { BadRequest, parseBool, parseDir, parseInteger, parsePaging } from "./errors";
import { mergeShards } from "./merge";

export const classesRoutes = new Hono<{ Bindings: Env }>();

export const LABELED_INLINE_MAX = 200;
const MEMBERS_PAGE = 100;
const MC_ROWID = sql`${mc}.rowid`;
const Q_ROWID = sql`${q}.rowid`;

// ---------------------------------------------------------------------------
// Browse list
// ---------------------------------------------------------------------------

const CLASS_SORT = {
  mc_id: mc.id,
  num_vertices: mc.n,
  class_size: mc.classSize,
  distinct_quiver_count: mc.distinctQuiverCount,
  dynkin_type: mc.dynkinType,
  class_type: mc.isOpen,
} as const;
type ClassSortKey = keyof typeof CLASS_SORT;

const CLASS_SELECTION = {
  rowid: sql<number>`${mc}.rowid`,
  id: mc.id, n: mc.n, label: mc.label, dynkinType: mc.dynkinType,
  isOpen: mc.isOpen, exploration: mc.exploration, classSize: mc.classSize,
  distinctQuiverCount: mc.distinctQuiverCount, mergedOrbitCount: mc.mergedOrbitCount,
  canonicalQuiverId: mc.canonicalQuiverId,
  isFiniteConfirmed: mc.isFiniteConfirmed, isInfiniteConfirmed: mc.isInfiniteConfirmed,
  isInfiniteExpected: mc.isInfiniteExpected,
  isMutationAcyclic: mc.isMutationAcyclic, isBanff: mc.isBanff,
  isLouise: mc.isLouise, isPPrime: mc.isPPrime,
  nickname: nick.nickname, nicknameSlug: nick.slug,
};
type ClassRow = Awaited<ReturnType<typeof selectClasses>>[number];

function selectClasses(db: Database) {
  return db.select(CLASS_SELECTION).from(mc).leftJoin(nick, eq(nick.mcId, mc.id));
}

export function classListItem(r: ClassRow) {
  return {
    mc_id: r.id,
    label: r.label,
    nickname: r.nickname,
    nickname_slug: r.nicknameSlug,
    num_vertices: r.n,
    dynkin_type: r.dynkinType,
    is_open: r.isOpen,
    exploration: r.exploration,
    class_size: r.isOpen ? null : r.classSize,
    labeled_size: r.classSize,
    distinct_quiver_count: r.distinctQuiverCount,
    merged_orbit_count: r.mergedOrbitCount,
    canonical_qid: r.canonicalQuiverId,
    is_finite_confirmed: r.isFiniteConfirmed,
    is_infinite_confirmed: r.isInfiniteConfirmed,
    is_infinite_expected: r.isInfiniteExpected,
    is_mutation_acyclic: r.isMutationAcyclic,
    is_banff: r.isBanff,
    is_louise: r.isLouise,
    is_p_prime: r.isPPrime,
  };
}

export interface ClassListParams {
  rank?: number; dynkinType?: string; isOpen?: boolean; isMutationFinite?: boolean;
  isMutationAcyclic?: boolean; orbitMin?: number; orbitMax?: number; nickname?: string;
  sort?: string; dir?: string; offset: number; limit: number; cursor?: string;
}

export function classListParamsFrom(get: (k: string) => string | undefined): ClassListParams {
  return {
    rank: parseInteger("rank", get("rank")),
    dynkinType: get("dynkin_type") || undefined,
    isOpen: parseBool("is_open", get("is_open")),
    isMutationFinite: parseBool("is_mutation_finite", get("is_mutation_finite")),
    isMutationAcyclic: parseBool("is_mutation_acyclic", get("is_mutation_acyclic")),
    orbitMin: parseInteger("orbit_min", get("orbit_min")),
    orbitMax: parseInteger("orbit_max", get("orbit_max")),
    nickname: get("nickname") || undefined,
    sort: get("sort"), dir: get("dir"), cursor: get("cursor"),
    ...parsePaging(get, 50),
  };
}

export async function listClasses(env: Env, p: ClassListParams) {
  const conds: SQL[] = [];
  if (p.rank !== undefined) conds.push(eq(mc.n, p.rank));
  if (p.dynkinType) conds.push(eq(mc.dynkinType, p.dynkinType));
  if (p.isOpen !== undefined) conds.push(eq(mc.isOpen, p.isOpen));
  if (p.isMutationFinite !== undefined) {
    conds.push(p.isMutationFinite ? eq(mc.isFiniteConfirmed, true) : eq(mc.isInfiniteConfirmed, true));
  }
  if (p.isMutationAcyclic !== undefined) conds.push(eq(mc.isMutationAcyclic, p.isMutationAcyclic));
  if (p.orbitMin !== undefined) conds.push(sql`${mc.classSize} >= ${p.orbitMin}`);
  if (p.orbitMax !== undefined) conds.push(sql`${mc.classSize} <= ${p.orbitMax}`);
  if (p.nickname) conds.push(eq(nick.slug, p.nickname.toLowerCase()));

  const sortKey = (p.sort ?? "num_vertices") as ClassSortKey;
  if (!Object.hasOwn(CLASS_SORT, sortKey)) {
    throw new BadRequest(`sort must be one of ${Object.keys(CLASS_SORT).join(", ")}`);
  }
  const dir = parseDir(p.dir);
  const cols: KeyCol[] = sortKey === "num_vertices" ? [mc.n, MC_ROWID]
    : sortKey === "mc_id" ? [mc.id] : [CLASS_SORT[sortKey], mc.n, MC_ROWID];
  const dirs: Dir[] = cols.map((_, i) => (i === 0 ? dir : "asc"));
  const where = conds.length ? and(...conds) : undefined;
  const shards = shardsForRank(p.rank);

  const onlyRank = conds.length === (p.rank !== undefined ? 1 : 0);
  const total = onlyRank
    ? (await mainDb(env).select().from(rankStats)
        .where(p.rank !== undefined ? eq(rankStats.n, p.rank) : undefined))
        .reduce((a, r) => a + r.classCount, 0)
    : (await Promise.all(shards.map((s) => dbOf(env, s).select({ n: sql<number>`count(*)` }).from(mc)
        .leftJoin(nick, eq(nick.mcId, mc.id)).where(where))))
        .reduce((a, r) => a + (r[0]?.n ?? 0), 0);

  const keyOf = (r: ClassRow): Key => cols.map((c) => c === mc.id ? r.id : c === mc.n ? r.n
    : c === mc.classSize ? r.classSize : c === mc.distinctQuiverCount ? r.distinctQuiverCount
    : c === mc.dynkinType ? r.dynkinType : c === mc.isOpen ? Number(r.isOpen) : r.rowid);
  const r = await mergeShards<ClassRow>({
    shardKeys: shards.map((s) => s.key), dirs, keyOf,
    fetch: (sk, after, limit) => selectClasses(dbOf(env, shards.find((s) => s.key === sk)!))
      .where(and(where, after ? afterKey(cols, dirs, after) : undefined))
      .orderBy(...orderBy(cols, dirs)).limit(limit),
    limit: p.limit, offset: p.offset, cursor: p.cursor,
  });
  return { items: r.items.map(classListItem), total, next_cursor: r.next_cursor };
}

classesRoutes.get("/", async (c) => {
  return c.json(await listClasses(c.env, classListParamsFrom((k) => c.req.query(k))));
});

// ---------------------------------------------------------------------------
// Members (paged): distinct quivers (all shards of the rank), labelings (class shard)
// ---------------------------------------------------------------------------

/**
 * Distinct quivers of a class, most-labeled first then rowid. Page 1 pins the
 * canonical quiver at the top; later pages exclude it. Key: [labeling_count, rowid].
 */
export async function classQuivers(env: Env, mcId: string, canonicalQid: string | null,
                                   cursor: string | undefined, limit: number) {
  const n = rankFromId(mcId) ?? 0;
  const shards = shardsForRank(n);
  const cols: KeyCol[] = [q.labelingCount, Q_ROWID];
  const dirs: Dir[] = ["desc", "asc"];
  type Row = { rowid: number; id: string; m: string; lc: number | null };
  const items: { qmd_id: string; matrix: number[][]; labeling_count: number; is_canonical: boolean }[] = [];
  let take = limit;
  if (!cursor && canonicalQid) {
    const cdb = dbForId(env, canonicalQid);
    const canon = cdb ? (await cdb.select({ id: q.id, m: q.exchangeMatrix, lc: q.labelingCount })
      .from(q).where(eq(q.id, canonicalQid)))[0] : undefined;
    if (canon) {
      items.push({ qmd_id: canon.id, matrix: decodeUpper(n, canon.m), labeling_count: canon.lc ?? 1, is_canonical: true });
      take -= 1;
    }
  }
  const r = await mergeShards<Row>({
    shardKeys: shards.map((s) => s.key), dirs,
    keyOf: (row) => [row.lc, row.rowid],
    fetch: (sk, after, lim) => dbOf(env, shards.find((s) => s.key === sk)!)
      .select({ rowid: sql<number>`${q}.rowid`, id: q.id, m: q.exchangeMatrix, lc: q.labelingCount })
      .from(q).where(and(eq(q.mutationClassId, mcId),
                         canonicalQid ? ne(q.id, canonicalQid) : undefined,
                         after ? afterKey(cols, dirs, after) : undefined))
      .orderBy(...orderBy(cols, dirs)).limit(lim),
    limit: take, cursor,
  });
  for (const row of r.items) {
    items.push({ qmd_id: row.id, matrix: decodeUpper(n, row.m), labeling_count: row.lc ?? 1, is_canonical: false });
  }
  return { items, next_cursor: r.next_cursor };
}

/** Labeled matrices of a class in orbit order (optionally one quiver's). Key: [ord]. */
export async function classLabelings(env: Env, mcId: string, qmdId: string | undefined,
                                     cursor: string | undefined, limit: number) {
  const db = dbForId(env, mcId);
  if (!db) return { items: [], next_cursor: null };
  const n = rankFromId(mcId) ?? 0;
  const after = decodeCursor(cursor, 1);
  const rows = await db.select({ ord: lab.ord, qmdId: lab.qmdId, matrix: lab.matrix })
    .from(lab)
    .where(and(eq(lab.mutationClassId, mcId),
               qmdId ? eq(lab.qmdId, qmdId) : undefined,
               after ? gt(lab.ord, after[0] as number) : undefined))
    .orderBy(lab.ord).limit(limit + 1);
  const page = rows.slice(0, limit);
  const last = page[page.length - 1];
  return {
    items: page.map((r) => ({ ord: r.ord, qmd_id: r.qmdId, matrix: decodeUpper(n, r.matrix) })),
    next_cursor: rows.length > limit && last ? encodeCursor([last.ord]) : null,
  };
}

// ---------------------------------------------------------------------------
// Detail
// ---------------------------------------------------------------------------

export async function classDetail(env: Env, id: string) {
  const db = dbForId(env, id);
  if (!db) return null;
  const row = (await db.select().from(mc).leftJoin(nick, eq(nick.mcId, mc.id)).where(eq(mc.id, id)))[0];
  if (!row) return null;
  const m = row.mutation_classes;
  const nn = row.class_nicknames;

  const distinct = await classQuivers(env, id, m.canonicalQuiverId, undefined, MEMBERS_PAGE);
  const inline = m.exploration === "complete" && m.classSize !== null && m.classSize <= LABELED_INLINE_MAX;
  const labeled = inline ? (await classLabelings(env, id, undefined, undefined, LABELED_INLINE_MAX)).items
    .map((r) => ({ qmd_id: r.qmd_id, matrix: r.matrix })) : [];

  return {
    mc_id: m.id,
    label: m.label,
    nickname: nn?.nickname ?? null,
    nickname_slug: nn?.slug ?? null,
    nickname_note: nn?.note ?? null,
    num_vertices: m.n,
    dynkin_type: m.dynkinType,
    is_open: m.isOpen,
    exploration: m.exploration,
    labeled_size: m.classSize,
    distinct_quiver_count: m.distinctQuiverCount,
    merged_orbit_count: m.mergedOrbitCount,
    canonical_matrix: decodeUpper(m.n, m.canonicalMatrix),
    canonical_qid: m.canonicalQuiverId,
    distinct_quivers: distinct.items,
    distinct_quivers_next_cursor: distinct.next_cursor,
    labeled_quivers: labeled,
    labelings_stored: m.classSize !== null,
    labeled_quivers_truncated: !inline,
    is_finite_confirmed: m.isFiniteConfirmed,
    is_infinite_confirmed: m.isInfiniteConfirmed,
    is_infinite_expected: m.isInfiniteExpected,
    size_of_explored_mutation_class: m.classSize,
    size_of_explored_frontier: m.sizeOfExploredFrontier,
    is_mutation_acyclic: m.isMutationAcyclic,
    is_banff: m.isBanff,
    is_louise: m.isLouise,
    is_p_prime: m.isPPrime,
    provenance: m.provenance,
  };
}

async function detailResponse(c: Context<{ Bindings: Env }>, id: string) {
  const d = await classDetail(c.env, id);
  if (!d) return c.json({ detail: "Mutation class not found" }, 404);
  return c.json(d);
}

/** Resolve a nickname slug to its class id (curated table, main database). */
export async function slugToId(env: Env, slug: string): Promise<string | null> {
  const r = (await mainDb(env).select({ id: nick.mcId }).from(nick).where(eq(nick.slug, slug.toLowerCase())))[0];
  return r?.id ?? null;
}

classesRoutes.get("/by-slug/:slug", async (c) => {
  const id = await slugToId(c.env, c.req.param("slug"));
  if (!id) return c.json({ detail: "No class with that nickname" }, 404);
  return detailResponse(c, id);
});

classesRoutes.get("/:id", (c) => detailResponse(c, c.req.param("id")));

classesRoutes.get("/:id/quivers", async (c) => {
  const id = c.req.param("id");
  const db = dbForId(c.env, id);
  if (!db) return c.json({ detail: "Mutation class not found" }, 404);
  const row = (await db.select({ canon: mc.canonicalQuiverId }).from(mc).where(eq(mc.id, id)))[0];
  if (!row) return c.json({ detail: "Mutation class not found" }, 404);
  const { limit } = parsePaging((k) => c.req.query(k), MEMBERS_PAGE);
  return c.json({ mc_id: id, ...(await classQuivers(c.env, id, row.canon, c.req.query("cursor"), limit)) });
});

classesRoutes.get("/:id/labelings", async (c) => {
  const id = c.req.param("id");
  const db = dbForId(c.env, id);
  if (!db) return c.json({ detail: "Mutation class not found" }, 404);
  const row = (await db.select({ id: mc.id }).from(mc).where(eq(mc.id, id)))[0];
  if (!row) return c.json({ detail: "Mutation class not found" }, 404);
  const { limit } = parsePaging((k) => c.req.query(k), MEMBERS_PAGE);
  return c.json({ mc_id: id,
    ...(await classLabelings(c.env, id, c.req.query("qmd_id") || undefined, c.req.query("cursor"), limit)) });
});
