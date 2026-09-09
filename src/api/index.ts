/**
 * API router, mounted at /api by src/index.ts.
 *
 * Routes and response shapes are driven by what the frontend calls
 * (public/*.html, download.js) and mirror the legacy FastAPI backend (git
 * history): /quivers, /quivers/{id}, /search, /classes/{id}, /export.
 * Added since: /stats, /classes (browse), /random/{quiver,class},
 * /export.csv, and — phase 2 — keyset cursors on every list,
 * /classes/{id}/quivers, /classes/{id}/labelings, /quivers/{id}/labelings,
 * /classes/by-slug/{slug}, /nicknames, /export.ndjson, /openapi.json, /lookup.
 *
 * Read-only public dataset: CORS is open, GET responses that only change on
 * ingest are edge-cacheable for 5 minutes, exports are never cached.
 */

import { asc } from "drizzle-orm";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { rankStats } from "../db/schema";
import { createPool, mainDb, withDb } from "../db/shard";
import { classesRoutes } from "./classes";
import { BadRequest, Unavailable } from "./errors";
import { exportRoutes } from "./export";
import { lookupRoutes } from "./lookup";
import { nicknamesRoutes } from "./nicknames";
import { openapiRoutes } from "./openapi";
import { listHandler, quiversRoutes } from "./quivers";
import { randomRoutes } from "./random";

export const api = new Hono<{ Bindings: Env }>();

api.use("*", cors({ origin: "*", allowMethods: ["GET", "HEAD", "OPTIONS"], maxAge: 86400 }));

// Default cache policy for data reads; handlers that must not be cached
// (exports, random) set their own Cache-Control.
api.use("*", async (c, next) => {
  await next();
  if (c.req.method === "GET" && c.res.status === 200 && !c.res.headers.has("Cache-Control")) {
    c.res.headers.set("Cache-Control", "public, max-age=300");
  }
});

/**
 * One Postgres pool per request, handed to the handlers on a shallow copy of
 * env, and closed after the response.
 *
 * `waitUntil` rather than an inline await: a streamed export finishes writing
 * after the handler returns, so ending the pool synchronously would cut the
 * stream. `/health` is exempt so that a liveness probe never opens a
 * connection -- it is what tells us the Worker is up when the database is not.
 *
 * A pool exhaustion or connect failure surfaces as `Unavailable` -> 503 with
 * Retry-After, never as an untyped 500. A PS-40 allows only 25 connections,
 * and an agent following /llms.txt must be able to tell "back off" from
 * "give up".
 */
api.use("*", async (c, next) => {
  if (c.req.path === "/api/health") return next();
  let pool;
  try {
    pool = createPool(c.env);
  } catch (e) {
    console.error("pool creation failed", e);
    throw new Unavailable("database is not reachable");
  }
  c.env = withDb(c.env, pool);
  const close = () => pool.end().catch((e) => console.error("pool.end failed", e));
  try {
    await next();
  } catch (e) {
    c.executionCtx.waitUntil(close());
    throw e;
  }
  // The pool must outlive the RESPONSE BODY, not the handler. /export and
  // /export.ndjson return a stream that is still pulling pages after the
  // handler has returned, so closing here on handler exit killed the export
  // mid-flight: the first page landed, the second failed with "Failed query",
  // and the client got a CSV containing nothing but its header row. Pipe the
  // body through a pass-through and close when that drains. A buffered JSON
  // response drains immediately, so this costs nothing on the common path.
  const body = c.res.body;
  if (body) {
    const { readable, writable } = new TransformStream();
    c.executionCtx.waitUntil(body.pipeTo(writable).finally(close));
    c.res = new Response(readable, c.res);
  } else {
    c.executionCtx.waitUntil(close());
  }
});

// Bad query params (unparseable ints/bools, unknown sort, bad cursor) -> 400.
api.onError((err, c) => {
  if (err instanceof BadRequest) return c.json({ detail: err.message }, 400);
  if (err instanceof Unavailable) {
    console.error(err);
    c.header("Retry-After", String(err.retryAfter));
    return c.json({ detail: err.message }, 503);
  }
  console.error(err);
  return c.json({ detail: "Internal server error" }, 500);
});
api.notFound((c) => c.json({ detail: "Not found" }, 404));

api.get("/health", (c) => {
  c.header("Cache-Control", "no-store");
  return c.json({ status: "ok" });
});

api.route("/quivers", quiversRoutes);
api.get("/search", listHandler(100));
api.route("/classes", classesRoutes);
api.route("/", exportRoutes);          // /export, /export.csv, /export.ndjson
api.route("/random", randomRoutes);
api.route("/nicknames", nicknamesRoutes);
api.route("/", openapiRoutes);         // /openapi.json
api.route("/", lookupRoutes);          // /lookup (matrix -> quiver)

/**
 * Homepage counts, served from the ingest-time aggregates table — never from
 * scans — plus how each rank was generated.
 */
api.get("/stats", async (c) => {
  const rows = await mainDb(c.env).select().from(rankStats).orderBy(asc(rankStats.n));
  const totals = rows.reduce(
    (acc, r) => ({
      quivers: acc.quivers + r.quiverCount,
      labeledQuivers: acc.labeledQuivers + r.labeledQuiverCount,
      classes: acc.classes + r.classCount,
    }),
    { quivers: 0, labeledQuivers: 0, classes: 0 },
  );
  return c.json({
    distinct_quivers: totals.quivers,
    labeled_quivers: totals.labeledQuivers,
    mutation_classes: totals.classes,
    by_rank: rows.map((r) => ({
      n: r.n,
      distinct_quivers: r.quiverCount,
      labeled_quivers: r.labeledQuiverCount,
      mutation_classes: r.classCount,
      bound: r.bound,
      node_cap: r.nodeCap,
      generated_at: r.generatedAt,
      pipeline_version: r.pipelineVersion,
      generator: r.generator,
      census_size: r.censusSize,
      shard_counts: r.shardCounts,
    })),
  });
});
