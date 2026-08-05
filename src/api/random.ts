/**
 * GET /random/quiver and /random/class — a uniformly random entity, found
 * via the ingest-time rank_stats counts + an indexed OFFSET (never
 * ORDER BY RANDOM(), which would scan).  Rank first, then offset within the
 * rank, matches the (n, id) index and routes cleanly through shardFor.
 */

import { asc, eq } from "drizzle-orm";
import { Hono } from "hono";
import { mutationClasses as mc, quivers as q, rankStats } from "../db/schema";
import { dbFor } from "../db/shard";

export const randomRoutes = new Hono<{ Bindings: Env }>();

async function pickRank(env: Env, kind: "quiver" | "class") {
  // Stats live in the index DB (shard 0 today == the only DB).
  const stats = await dbFor(env, 0).select().from(rankStats)
    .orderBy(asc(rankStats.n));
  const counts = stats.map((s) => ({
    n: s.n,
    count: kind === "quiver" ? s.quiverCount : s.classCount,
  }));
  const total = counts.reduce((acc, c) => acc + c.count, 0);
  if (total === 0) return null;
  let r = Math.floor(Math.random() * total);
  for (const c of counts) {
    if (r < c.count) return { n: c.n, offset: r };
    r -= c.count;
  }
  return null;   // unreachable
}

randomRoutes.get("/quiver", async (c) => {
  const pick = await pickRank(c.env, "quiver");
  if (!pick) return c.json({ detail: "Database is empty" }, 404);
  const row = (await dbFor(c.env, pick.n)
    .select({ id: q.id, n: q.n }).from(q).where(eq(q.n, pick.n))
    .orderBy(asc(q.id)).offset(pick.offset).limit(1))[0];
  if (!row) return c.json({ detail: "Database is empty" }, 404);
  c.header("Cache-Control", "no-store");
  return c.json({ qmd_id: row.id, num_vertices: row.n });
});

randomRoutes.get("/class", async (c) => {
  const pick = await pickRank(c.env, "class");
  if (!pick) return c.json({ detail: "Database is empty" }, 404);
  const row = (await dbFor(c.env, pick.n)
    .select({ id: mc.id, n: mc.n }).from(mc).where(eq(mc.n, pick.n))
    .orderBy(asc(mc.id)).offset(pick.offset).limit(1))[0];
  if (!row) return c.json({ detail: "Database is empty" }, 404);
  c.header("Cache-Control", "no-store");
  return c.json({ mc_id: row.id, num_vertices: row.n });
});
