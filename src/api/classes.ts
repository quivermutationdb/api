/**
 * /classes routes: browse list, detail, paged members, nickname lookup.
 *
 * The detail shape mirrors the legacy ClassDetail (the frontend's contract)
 * with these phase-2 changes (docs/PHASE2.md §4): the orbit is no longer
 * embedded — `distinct_quivers` is the first page (canonical quiver first)
 * with `distinct_quivers_next_cursor`, and `labeled_quivers` is present only
 * for small classes (<= LABELED_INLINE_MAX rows), otherwise
 * `labeled_quivers_truncated: true`. Use /classes/{id}/quivers and
 * /classes/{id}/labelings to page through the rest.
 */

import { and, desc, eq, gt, ne, sql, type SQL } from "drizzle-orm";
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

export const classesRoutes = new Hono<{ Bindings: Env }>();

export const LABELED_INLINE_MAX = 200;
const MEMBERS_PAGE = 100;

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

export async function listClasses(db: Database, p: ClassListParams) {
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
  const cols = sortKey === "num_vertices" ? [mc.n, mc.id]
    : sortKey === "mc_id" ? [mc.id] : [CLASS_SORT[sortKey], mc.n, mc.id];
  const dirs: ("asc" | "desc")[] = cols.map((_, i) => (i === 0 ? dir : "asc"));
  const after = decodeCursor(p.cursor, cols.length);

  const onlyRank = conds.length === (p.rank !== undefined ? 1 : 0);
  const total = onlyRank
    ? (await db.select().from(rankStats)
        .where(p.rank !== undefined ? eq(rankStats.n, p.rank) : undefined))
        .reduce((a, r) => a + r.classCount, 0)
    : ((await db.select({ n: sql<number>`count(*)` }).from(mc)
        .leftJoin(nick, eq(nick.mcId, mc.id))
        .where(conds.length ? and(...conds) : undefined))[0]?.n ?? 0);

  const rows = await selectClasses(db)
    .where(and(...conds, after ? afterKey(cols, dirs, after) : undefined))
    .orderBy(...orderBy(cols, dirs))
    .offset(after ? 0 : p.offset).limit(p.limit + 1);
  const page = rows.slice(0, p.limit);
  const last = page[page.length - 1];
  const keyOf = (r: ClassRow) => cols.map((c) => c === mc.id ? r.id : c === mc.n ? r.n
    : c === mc.classSize ? r.classSize : c === mc.distinctQuiverCount ? r.distinctQuiverCount
    : c === mc.dynkinType ? r.dynkinType : Number(r.isOpen));
  return {
    items: page.map(classListItem),
    total,
    next_cursor: rows.length > p.limit && last ? encodeCursor(keyOf(last)) : null,
  };
}

classesRoutes.get("/", async (c) => {
  const p = classListParamsFrom((k) => c.req.query(k));
  return c.json(await listClasses(dbFor(c.env, p.rank ?? 0), p));
});

// ---------------------------------------------------------------------------
// Members (paged): distinct quivers, labelings
// ---------------------------------------------------------------------------

/**
 * Distinct quivers of a class, most-labeled first then id. Page 1 pins the
 * canonical quiver at the top; later pages exclude it. Keyset key:
 * [labeling_count, id].
 */
export async function classQuivers(db: Database, mcId: string, canonicalQid: string | null,
                                   cursor: string | undefined, limit: number) {
  const cols = [q.labelingCount, q.id];
  const dirs: ("asc" | "desc")[] = ["desc", "asc"];
  const after = decodeCursor(cursor, 2);
  const conds = [eq(q.mutationClassId, mcId)];
  if (canonicalQid) conds.push(ne(q.id, canonicalQid));
  if (after) conds.push(afterKey(cols, dirs, after));

  const items: { qmd_id: string; matrix: Matrix; labeling_count: number; is_canonical: boolean }[] = [];
  let take = limit;
  if (!after && canonicalQid) {
    const canon = (await db.select({ id: q.id, m: q.exchangeMatrix, lc: q.labelingCount })
      .from(q).where(eq(q.id, canonicalQid)))[0];
    if (canon) {
      items.push({ qmd_id: canon.id, matrix: canon.m, labeling_count: canon.lc ?? 1, is_canonical: true });
      take -= 1;
    }
  }
  const rows = await db.select({ id: q.id, m: q.exchangeMatrix, lc: q.labelingCount })
    .from(q).where(and(...conds)).orderBy(desc(q.labelingCount), q.id).limit(take + 1);
  for (const r of rows.slice(0, take)) {
    items.push({ qmd_id: r.id, matrix: r.m, labeling_count: r.lc ?? 1, is_canonical: false });
  }
  const last = rows[Math.min(take, rows.length) - 1];
  return {
    items,
    next_cursor: rows.length > take && last ? encodeCursor([last.lc ?? 1, last.id]) : null,
  };
}

/** Labeled matrices of a class in orbit order (optionally one quiver's). Key: [ord]. */
export async function classLabelings(db: Database, mcId: string, qmdId: string | undefined,
                                     cursor: string | undefined, limit: number) {
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
    items: page.map((r) => ({ ord: r.ord, qmd_id: r.qmdId, matrix: r.matrix })),
    next_cursor: rows.length > limit && last ? encodeCursor([last.ord]) : null,
  };
}

// ---------------------------------------------------------------------------
// Detail
// ---------------------------------------------------------------------------

export async function classDetail(db: Database, id: string) {
  const row = (await db.select().from(mc).leftJoin(nick, eq(nick.mcId, mc.id))
    .where(eq(mc.id, id)))[0];
  if (!row) return null;
  const m = row.mutation_classes;
  const nn = row.class_nicknames;

  const distinct = await classQuivers(db, id, m.canonicalQuiverId, undefined, MEMBERS_PAGE);
  const inline = m.classSize <= LABELED_INLINE_MAX;
  const labeled = inline ? (await classLabelings(db, id, undefined, undefined, LABELED_INLINE_MAX)).items
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
    canonical_matrix: m.canonicalMatrix,
    canonical_qid: m.canonicalQuiverId,
    distinct_quivers: distinct.items,
    distinct_quivers_next_cursor: distinct.next_cursor,
    labeled_quivers: labeled,
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
  const db = dbForId(c.env, id);
  const d = db ? await classDetail(db, id) : null;
  if (!d) return c.json({ detail: "Mutation class not found" }, 404);
  return c.json(d);
}

/** Resolve a nickname slug to its class id (slugs are global, so the index DB answers). */
export async function slugToId(db: Database, slug: string): Promise<string | null> {
  const r = (await db.select({ id: nick.mcId }).from(nick).where(eq(nick.slug, slug.toLowerCase())))[0];
  return r?.id ?? null;
}

classesRoutes.get("/by-slug/:slug", async (c) => {
  const id = await slugToId(dbFor(c.env, 0), c.req.param("slug"));
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
  return c.json({ mc_id: id, ...(await classQuivers(db, id, row.canon, c.req.query("cursor"), limit)) });
});

classesRoutes.get("/:id/labelings", async (c) => {
  const id = c.req.param("id");
  const db = dbForId(c.env, id);
  if (!db) return c.json({ detail: "Mutation class not found" }, 404);
  const row = (await db.select({ id: mc.id }).from(mc).where(eq(mc.id, id)))[0];
  if (!row) return c.json({ detail: "Mutation class not found" }, 404);
  const { limit } = parsePaging((k) => c.req.query(k), MEMBERS_PAGE);
  return c.json({ mc_id: id,
    ...(await classLabelings(db, id, c.req.query("qmd_id") || undefined, c.req.query("cursor"), limit)) });
});
