/**
 * Mutation-class routes: browse list + detail.
 *
 * The detail shape mirrors the legacy FastAPI backend's ClassDetail (see git
 * history: qmd/schemas.py, qmd/crud.py) — the frontend's contract.
 * distinct_quivers collapses the labeled orbit to one entry per unlabeled
 * quiver; each entry's matrix is the quiver's *canonical form*, which in the
 * Python backend was recomputed with canonical_form() — here it is looked up
 * from the quivers table instead (same value, no matrix math in the Worker).
 */

import { and, asc, desc, eq, inArray, sql, type SQL } from "drizzle-orm";
import { Hono } from "hono";
import {
  mutationClasses as mc,
  mutationClassPayloads as payloads,
  quivers as q,
  type Matrix,
} from "../db/schema";
import { dbFor, dbForId } from "../db/shard";
import { BadRequest } from "./quivers";

export const classesRoutes = new Hono<{ Bindings: Env }>();

// ---------------------------------------------------------------------------
// Browse list (new in the Cloudflare API; flat and stable)
// ---------------------------------------------------------------------------

const CLASS_SORT = {
  mc_id: mc.id,
  num_vertices: mc.n,
  class_size: mc.classSize,
  distinct_quiver_count: mc.distinctQuiverCount,
  dynkin_type: mc.dynkinType,
  class_type: mc.isOpen,
} as const;

classesRoutes.get("/", async (c) => {
  const get = (k: string) => c.req.query(k);
  const int = (k: string) => {
    const v = get(k);
    if (v === undefined || v === "") return undefined;
    const i = Number(v);
    if (!Number.isInteger(i)) throw new BadRequest(`${k} must be an integer`);
    return i;
  };
  const bool = (k: string) => {
    const v = get(k)?.toLowerCase();
    if (v === undefined || v === "") return undefined;
    if (v === "true" || v === "1") return true;
    if (v === "false" || v === "0") return false;
    throw new BadRequest(`${k} must be true or false`);
  };

  const conds: SQL[] = [];
  const rank = int("rank");
  if (rank !== undefined) conds.push(eq(mc.n, rank));
  const dynkin = get("dynkin_type");
  if (dynkin) conds.push(eq(mc.dynkinType, dynkin));
  const isOpen = bool("is_open");
  if (isOpen !== undefined) conds.push(eq(mc.isOpen, isOpen));
  const mutFinite = bool("is_mutation_finite");
  if (mutFinite !== undefined) {
    conds.push(mutFinite ? eq(mc.isFiniteConfirmed, true) : eq(mc.isInfiniteConfirmed, true));
  }
  const mutAcyclic = bool("is_mutation_acyclic");
  if (mutAcyclic !== undefined) conds.push(eq(mc.isMutationAcyclic, mutAcyclic));
  const sizeMin = int("orbit_min");
  if (sizeMin !== undefined) conds.push(sql`${mc.classSize} >= ${sizeMin}`);
  const sizeMax = int("orbit_max");
  if (sizeMax !== undefined) conds.push(sql`${mc.classSize} <= ${sizeMax}`);
  const where = conds.length ? and(...conds) : undefined;

  const sortKey = get("sort") ?? "num_vertices";
  if (!Object.hasOwn(CLASS_SORT, sortKey)) {
    throw new BadRequest(`sort must be one of ${Object.keys(CLASS_SORT).join(", ")}`);
  }
  const dir = get("dir");
  if (dir !== undefined && dir !== "asc" && dir !== "desc") {
    throw new BadRequest("dir must be 'asc' or 'desc'");
  }
  const col = CLASS_SORT[sortKey as keyof typeof CLASS_SORT];
  const order = [dir === "desc" ? desc(col) : asc(col), asc(mc.id)];
  const offset = Math.max(int("offset") ?? 0, 0);
  const limit = Math.min(Math.max(int("limit") ?? 50, 1), 1000);

  const db = dbFor(c.env, rank ?? 0);
  const total = (await db.select({ n: sql<number>`count(*)` })
    .from(mc).where(where))[0]?.n ?? 0;
  const rows = await db.select().from(mc).where(where)
    .orderBy(...order).offset(offset).limit(limit);

  return c.json({
    items: rows.map((r) => ({
      mc_id: r.id,
      label: r.label,
      num_vertices: r.n,
      dynkin_type: r.dynkinType,
      is_open: r.isOpen,
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
    })),
    total,
  });
});

// ---------------------------------------------------------------------------
// Detail
// ---------------------------------------------------------------------------

classesRoutes.get("/:id", async (c) => {
  const id = c.req.param("id");
  const db = dbForId(c.env, id);
  if (!db) return c.json({ detail: "Mutation class not found" }, 404);

  const row = (await db.select().from(mc).where(eq(mc.id, id)))[0];
  if (!row) return c.json({ detail: "Mutation class not found" }, 404);

  const payload = (await db.select().from(payloads)
    .where(eq(payloads.mutationClassId, id)))[0];
  const labeled = payload?.labeledQuivers ?? [];

  // Collapse the labeled orbit to one entry per distinct unlabeled quiver,
  // in first-appearance order (as the legacy backend did).
  const groups = new Map<string, { count: number; fallback: Matrix }>();
  for (const e of labeled) {
    const g = groups.get(e.qmd_id);
    if (g) g.count += 1;
    else groups.set(e.qmd_id, { count: 1, fallback: e.matrix });
  }

  // Canonical form per quiver comes from the quivers table (identical to the
  // Python canonical_form() result — it is how the row was written).
  const qids = [...groups.keys()];
  const canonical = new Map<string, Matrix>();
  for (let i = 0; i < qids.length; i += 50) {
    const batch = await db.select({ id: q.id, m: q.exchangeMatrix })
      .from(q).where(inArray(q.id, qids.slice(i, i + 50)));
    for (const b of batch) canonical.set(b.id, b.m);
  }

  const distinct = qids.map((qid) => ({
    qmd_id: qid,
    matrix: canonical.get(qid) ?? groups.get(qid)!.fallback,
    labeling_count: groups.get(qid)!.count,
    is_canonical: qid === row.canonicalQuiverId,
  }));
  // Canonical first, then most-labeled, then id for a stable order.
  distinct.sort((a, b) =>
    Number(b.is_canonical) - Number(a.is_canonical)
    || b.labeling_count - a.labeling_count
    || (a.qmd_id < b.qmd_id ? -1 : a.qmd_id > b.qmd_id ? 1 : 0));

  // Shape: the legacy ClassDetail response, field for field.
  return c.json({
    mc_id: row.id,
    label: row.label,
    num_vertices: row.n,
    dynkin_type: row.dynkinType,
    is_open: row.isOpen,
    labeled_size: row.classSize,
    distinct_quiver_count: row.distinctQuiverCount,
    merged_orbit_count: row.mergedOrbitCount,
    canonical_matrix: row.canonicalMatrix,
    canonical_qid: row.canonicalQuiverId,
    distinct_quivers: distinct,
    labeled_quivers: labeled,
    is_finite_confirmed: row.isFiniteConfirmed,
    is_infinite_confirmed: row.isInfiniteConfirmed,
    is_infinite_expected: row.isInfiniteExpected,
    size_of_explored_mutation_class: row.classSize,
    size_of_explored_frontier: row.sizeOfExploredFrontier,
    is_mutation_acyclic: row.isMutationAcyclic,
    is_banff: row.isBanff,
    is_louise: row.isLouise,
    is_p_prime: row.isPPrime,
    provenance: row.provenance,
  });
});
