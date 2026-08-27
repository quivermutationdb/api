/**
 * GET /random/quiver and /random/class — a uniformly random entity, found
 * via the ingest-time rank_stats counts + an indexed OFFSET (never
 * ORDER BY RANDOM()). For a split rank, the shard is chosen in proportion to
 * its stored row count (rank_stats.shard_counts).
 */

import { asc, eq } from "drizzle-orm";
import { Hono } from "hono";
import { mutationClasses as mc, quivers as q, rankStats } from "../db/schema";
import { dbOf, mainDb, shardsForRank } from "../db/shard";

export const randomRoutes = new Hono<{ Bindings: Env }>();

async function pick(env: Env, kind: "quiver" | "class") {
  const stats = await mainDb(env).select().from(rankStats).orderBy(asc(rankStats.n));
  const counts = stats.map((s) => ({ n: s.n, count: kind === "quiver" ? s.quiverCount : s.classCount,
                                      shardCounts: s.shardCounts }));
  const total = counts.reduce((acc, c) => acc + c.count, 0);
  if (total === 0) return null;
  let r = Math.floor(Math.random() * total);
  for (const c of counts) {
    if (r < c.count) {
      const shards = shardsForRank(c.n);
      if (shards.length === 1) return { shard: shards[0]!, offset: r };
      // Distribute the offset across shards by their stored counts.
      const per = shards.map((s) => (c.shardCounts?.[s.key]?.[kind === "quiver" ? "quivers" : "classes"]) ?? 0);
      for (let i = 0; i < shards.length; i++) {
        if (r < per[i]!) return { shard: shards[i]!, offset: r };
        r -= per[i]!;
      }
      return { shard: shards[0]!, offset: 0 };
    }
    r -= c.count;
  }
  return null;
}

randomRoutes.get("/quiver", async (c) => {
  const p = await pick(c.env, "quiver");
  if (!p) return c.json({ detail: "Database is empty" }, 404);
  const row = (await dbOf(c.env, p.shard).select({ id: q.id, n: q.n }).from(q)
    .where(p.shard.rank !== undefined ? eq(q.n, p.shard.rank) : undefined)
    .orderBy(asc(q.n), asc(q.id)).offset(p.offset).limit(1))[0];
  if (!row) return c.json({ detail: "rank_stats out of sync with the rows" }, 404);
  c.header("Cache-Control", "no-store");
  return c.json({ qmd_id: row.id, num_vertices: row.n });
});

randomRoutes.get("/class", async (c) => {
  const p = await pick(c.env, "class");
  if (!p) return c.json({ detail: "Database is empty" }, 404);
  const row = (await dbOf(c.env, p.shard).select({ id: mc.id, n: mc.n }).from(mc)
    .where(p.shard.rank !== undefined ? eq(mc.n, p.shard.rank) : undefined)
    .orderBy(asc(mc.n), asc(mc.id)).offset(p.offset).limit(1))[0];
  if (!row) return c.json({ detail: "rank_stats out of sync with the rows" }, 404);
  c.header("Cache-Control", "no-store");
  return c.json({ mc_id: row.id, num_vertices: row.n });
});
