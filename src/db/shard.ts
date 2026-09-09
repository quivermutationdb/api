/**
 * The single routing seam for all database access.
 *
 * There is now ONE database: PlanetScale Postgres, reached through Hyperdrive.
 * The seam survives the collapse from sharded D1 because CLAUDE.md requires it
 * -- a future per-rank split changes only this module -- but every function
 * here is now a one-liner over a single handle.
 *
 * CONNECTION LIFECYCLE. Workers get one Pool per REQUEST, closed by
 * `ctx.waitUntil(pool.end())` in the middleware (src/api/index.ts). It is a
 * Pool rather than a Client because `pg`'s Client serialises: exactly one query
 * at a time, and a second concurrent query on the same Client throws rather
 * than queueing. listQuivers deliberately runs its count alongside its page
 * read, so a Client would have turned that optimisation into a runtime error.
 * Hyperdrive does the real pooling upstream; `max` here only bounds how many
 * of its multiplexed connections one request may hold at once, which matters
 * because a PS-40 caps out at max_connections = 25.
 */

import { drizzle, type NodePgDatabase } from "drizzle-orm/node-postgres";
import { Pool } from "pg";
import * as schema from "./schema";

export type Database = NodePgDatabase<typeof schema>;

/** Per-request handle, attached to a shallow copy of env by `withDb`. */
const DB_KEY = "__qmdDb" as const;

/** Bound above the 3 concurrent reads any single handler issues. */
const MAX_CONNECTIONS_PER_REQUEST = 3;

export function createPool(env: Env): Pool {
  const connectionString = env.HYPERDRIVE?.connectionString;
  if (!connectionString) throw new Error("HYPERDRIVE binding is not configured");
  const pool = new Pool({ connectionString, max: MAX_CONNECTIONS_PER_REQUEST });
  // MANDATORY, not defensive. `pg.Pool` emits "error" when a pooled socket dies
  // outside a query -- which is exactly what the server-side statement_timeout
  // causes: it cancels the statement and the connection goes away underneath
  // us. Node treats an unhandled "error" event as fatal, so without this a
  // single slow query killed the whole isolate with "This socket has been ended
  // by the other party", taking every concurrent request down with it. Observed,
  // not theorised. The request's own error path reports the failure; this
  // listener only has to keep the process alive.
  pool.on("error", (e) => console.error("idle pool client error", e));
  return pool;
}

/**
 * A shallow copy of env carrying the request's database handle. A COPY, never a
 * mutation: `env` is shared across every request in the isolate, so writing the
 * handle onto it would hand one request's pool to another.
 */
export function withDb(env: Env, pool: Pool): Env {
  return { ...env, [DB_KEY]: drizzle(pool, { schema }) } as Env;
}

function handle(env: Env): Database {
  const db = (env as unknown as Record<string, Database | undefined>)[DB_KEY];
  if (!db) throw new Error("no database handle on env: withDb middleware did not run");
  return db;
}

/**
 * The routing seam. One database today, so the rank is ignored; a future split
 * dispatches on it here and nowhere else.
 */
export function shardFor(_n?: number): (env: Env) => Database {
  return handle;
}

export function db(env: Env): Database {
  return handle(env);
}

/** Global tables (rank_stats, class_nicknames, downloads) and everything else. */
export const mainDb = handle;

/**
 * Rank encoded in a `Q.n{k}.{hash}` / `MC.n{k}.{hash}` id, or null if the id is
 * malformed. Still the cheap way to reject junk before it reaches the database.
 */
export function rankFromId(id: string): number | null {
  const m = /^(?:Q|MC)\.n(\d+)\.[0-9a-f]{16}$/.exec(id);
  return m ? Number(m[1]) : null;
}

/** Handle for the database holding `id`, or null when the id is malformed. */
export function dbForId(env: Env, id: string): Database | null {
  return rankFromId(id) === null ? null : handle(env);
}
