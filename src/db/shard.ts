/**
 * The single routing seam for all database access.
 *
 * Today there is one D1 database and `shardFor` returns it for every `n`.
 * When per-`n` shards are introduced (a future, human-approved step), only
 * this module changes: `shardFor` maps `n` to the right binding, the skinny
 * browse tables stay in a global index DB, and point lookups keep routing by
 * the rank encoded in the id prefix (`Q.n4.{hash}`, `MC.n4.{hash}`).
 */

import { drizzle, type DrizzleD1Database } from "drizzle-orm/d1";
import * as schema from "./schema";

export type Database = DrizzleD1Database<typeof schema>;

/** D1 binding that stores rank-`n` data. Currently the one bound DB for every `n`. */
export function shardFor(env: Env, _n: number): D1Database {
  return env.DB;
}

/** Drizzle handle over the shard that stores rank-`n` data. */
export function dbFor(env: Env, n: number): Database {
  return drizzle(shardFor(env, n), { schema });
}

/**
 * Rank encoded in a `Q.n{k}.{hash}` / `MC.n{k}.{hash}` id, or null if the id
 * is malformed. Point lookups parse this to route through `shardFor`.
 */
export function rankFromId(id: string): number | null {
  const m = /^(?:Q|MC)\.n(\d+)\./.exec(id);
  return m ? Number(m[1]) : null;
}

/** Drizzle handle for the shard holding `id`, or null for a malformed id. */
export function dbForId(env: Env, id: string): Database | null {
  const n = rankFromId(id);
  return n === null ? null : dbFor(env, n);
}
