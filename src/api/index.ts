/**
 * API router, mounted at /api by src/index.ts.
 *
 * Routes and response shapes are driven by what the live frontend calls
 * (browse.html, search.html, quiver.html, class.html, index.html,
 * download.js in the website repo) and mirror the FastAPI backend
 * (main.py + qmd/schemas.py): /quivers, /quivers/{id}, /search,
 * /classes/{id}, /export.  New, brief-mandated additions: /stats,
 * /classes (browse), /random/quiver, /random/class, /export.csv.
 */

import { asc } from "drizzle-orm";
import { Hono } from "hono";
import { rankStats } from "../db/schema";
import { dbFor } from "../db/shard";
import { classesRoutes } from "./classes";
import { exportRoutes } from "./export";
import { BadRequest, listHandler, quiversRoutes } from "./quivers";
import { randomRoutes } from "./random";

export const api = new Hono<{ Bindings: Env }>();

// Bad query params (unparseable ints/bools) -> 400, FastAPI-style detail.
api.onError((err, c) => {
  if (err instanceof BadRequest) return c.json({ detail: err.message }, 400);
  console.error(err);
  return c.json({ detail: "Internal server error" }, 500);
});

api.get("/health", (c) => c.json({ status: "ok" }));

api.route("/quivers", quiversRoutes);
api.get("/search", listHandler(100));
api.route("/classes", classesRoutes);
api.route("/", exportRoutes);          // /export and /export.csv
api.route("/random", randomRoutes);

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
