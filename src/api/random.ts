/**
 * GET /random/quiver and /random/class — a uniformly random entity, found via
 * the ingest-time rank_stats counts plus a single indexed seek on `seq`
 * (never ORDER BY RANDOM(), and never OFFSET — see `pick` below).
 */

import { and, asc, eq } from "drizzle-orm";
import { Hono } from "hono";
import { mutationClasses as mc, quivers as q, rankStats } from "../db/schema";
import { mainDb } from "../db/shard";

export const randomRoutes = new Hono<{ Bindings: Env }>();

/**
 * Pick a rank weighted by its row count, then a position within it.
 *
 * The position is a `seq` value, NOT an OFFSET. `seq` is assigned at load by
 * `row_number() OVER (PARTITION BY n ORDER BY id)` and verified gapless
 * 1..count at every rank, so `WHERE n = k AND seq = s` is an index seek on
 * idx_q_n_seq and hits exactly the row OFFSET would have walked to.
 *
 * The D1 version paged with OFFSET, which is survivable across four shards of
 * a 10 M-row rank and is not survivable against one 42.5 M-row table: measured
 * on this database, `ORDER BY n, id OFFSET 21000000 LIMIT 1` takes **82.7 s**
 * against **3.3 ms** for the seq seek. Never reintroduce the OFFSET form.
 */
async function pick(env: Env, kind: "quiver" | "class") {
  const stats = await mainDb(env).select().from(rankStats).orderBy(asc(rankStats.n));
  const counts = stats.map((s) => ({ n: s.n, count: Number(kind === "quiver" ? s.quiverCount : s.classCount) }));
  const total = counts.reduce((acc, c) => acc + c.count, 0);
  if (total === 0) return null;
  let r = Math.floor(Math.random() * total);
  for (const c of counts) {
    if (r < c.count) return { n: c.n, seq: r + 1 };   // seq is 1-based
    r -= c.count;
  }
  return null;
}

randomRoutes.get("/quiver", async (c) => {
  const p = await pick(c.env, "quiver");
  if (!p) return c.json({ detail: "Database is empty" }, 404);
  const row = (await mainDb(c.env).select({ id: q.id, n: q.n }).from(q)
    .where(and(eq(q.n, p.n), eq(q.seq, p.seq))).limit(1))[0];
  if (!row) return c.json({ detail: "rank_stats out of sync with the rows" }, 404);
  c.header("Cache-Control", "no-store");
  return c.json({ qmd_id: row.id, num_vertices: row.n });
});

randomRoutes.get("/class", async (c) => {
  const p = await pick(c.env, "class");
  if (!p) return c.json({ detail: "Database is empty" }, 404);
  const row = (await mainDb(c.env).select({ id: mc.id, n: mc.n }).from(mc)
    .where(and(eq(mc.n, p.n), eq(mc.seq, p.seq))).limit(1))[0];
  if (!row) return c.json({ detail: "rank_stats out of sync with the rows" }, 404);
  c.header("Cache-Control", "no-store");
  return c.json({ mc_id: row.id, num_vertices: row.n });
});
