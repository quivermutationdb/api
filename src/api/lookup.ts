/**
 * GET /lookup?matrix=[[0,1],[-1,0]]  (or POST {matrix}) — find the quiver a
 * pasted exchange matrix represents: canonicalise (lex-min), hash to its id,
 * and return the row if the census contains it. Never a guess: the response
 * always carries the canonical id, and `found: false` when it is absent.
 */

import { Hono } from "hono";
import { isConnected, isSkewSymmetric, lexminForm, quiverId } from "../canon";
import type { Matrix } from "../db/schema";
import { BadRequest } from "./errors";
import { quiverDetail } from "./quivers";

export const lookupRoutes = new Hono<{ Bindings: Env }>();

const MAX_RANK = 12;

export function parseMatrix(raw: unknown): Matrix {
  let m: unknown = raw;
  if (typeof raw === "string") {
    try { m = JSON.parse(raw); } catch { throw new BadRequest("matrix must be JSON like [[0,1],[-1,0]]"); }
  }
  if (!Array.isArray(m) || m.length === 0 || m.length > MAX_RANK
      || !m.every((r) => Array.isArray(r) && r.every((x) => Number.isInteger(x) && Math.abs(x as number) <= 1000))) {
    throw new BadRequest(`matrix must be a square integer matrix with 1..${MAX_RANK} rows`);
  }
  const mat = m as Matrix;
  if (!isSkewSymmetric(mat)) throw new BadRequest("matrix must be skew-symmetric (b_ji = -b_ij, zero diagonal)");
  return mat;
}

export async function lookupMatrix(env: Env, raw: unknown) {
  const m = parseMatrix(raw);
  const canonical = lexminForm(m);
  const id = await quiverId(canonical);
  const quiver = await quiverDetail(env, id);
  return {
    qmd_id: id,
    num_vertices: m.length,
    canonical_matrix: canonical,
    is_connected: isConnected(m),
    max_edge: Math.max(0, ...m.flat().map(Math.abs)),
    found: quiver !== null,
    quiver,
  };
}

lookupRoutes.get("/lookup", async (c) => {
  const raw = c.req.query("matrix");
  if (!raw) throw new BadRequest("matrix query parameter required");
  c.header("Cache-Control", "public, max-age=300");
  return c.json(await lookupMatrix(c.env, raw));
});

lookupRoutes.post("/lookup", async (c) => {
  const body = await c.req.json().catch(() => { throw new BadRequest("JSON body required: {\"matrix\": [[...]]}"); });
  return c.json(await lookupMatrix(c.env, (body as { matrix?: unknown }).matrix));
});
