/**
 * API router, mounted at /api by src/index.ts.
 *
 * Endpoints are implemented in migration step 4 (driven by what the live
 * frontend calls, preserving existing response shapes). This scaffold wires
 * up the router, the DB access path (Drizzle over `shardFor`), and the two
 * endpoints that need no frontend-shape verification: /health and /stats.
 */

import { asc } from "drizzle-orm";
import { Hono } from "hono";
import { rankStats } from "../db/schema";
import { dbFor } from "../db/shard";

export const api = new Hono<{ Bindings: Env }>();

api.get("/health", (c) => c.json({ status: "ok" }));

/**
 * Homepage counts, served from the ingest-time aggregates table — never from
 * scans. Edge-cacheable: the data only changes on (re-)ingest.
 */
api.get("/stats", async (c) => {
  // Aggregates live alongside the skinny browse tables (rank 0 shard = the
  // global/index DB once sharding exists).
  const db = dbFor(c.env, 0);
  const rows = await db
    .select()
    .from(rankStats)
    .orderBy(asc(rankStats.n));

  const totals = rows.reduce(
    (acc, r) => ({
      quivers: acc.quivers + r.quiverCount,
      labeledQuivers: acc.labeledQuivers + r.labeledQuiverCount,
      classes: acc.classes + r.classCount,
    }),
    { quivers: 0, labeledQuivers: 0, classes: 0 },
  );

  c.header("Cache-Control", "public, max-age=300");
  return c.json({
    distinct_quivers: totals.quivers,
    labeled_quivers: totals.labeledQuivers,
    mutation_classes: totals.classes,
    by_rank: rows.map((r) => ({
      n: r.n,
      distinct_quivers: r.quiverCount,
      labeled_quivers: r.labeledQuiverCount,
      mutation_classes: r.classCount,
    })),
  });
});
