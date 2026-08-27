/**
 * The single routing seam for all database access.
 *
 * Ranks live in the main database except those listed in data/shards.json,
 * which are split across several databases by the bucket of the id hash
 * (bucket = first hex digit of the hash mod buckets). Point lookups route by
 * id; lists query every shard that can hold matching rows and merge
 * (src/api/merge.ts). Global tables (rank_stats, class_nicknames, downloads)
 * live in the main database only.
 */

import { drizzle, type DrizzleD1Database } from "drizzle-orm/d1";
import * as schema from "./schema";
import shardsConfig from "../../data/shards.json";

export type Database = DrizzleD1Database<typeof schema>;

export interface Shard {
  /** Stable key used in composite cursors and logs, e.g. "main", "n6.0". */
  key: string;
  binding: string;
  /** Rank this shard is dedicated to (undefined = the main database). */
  rank?: number;
  bucket?: number;
}

const SPLIT: Record<string, { buckets: number; databases: { binding: string }[] }> = shardsConfig.split;

export const MAIN_SHARD: Shard = { key: "main", binding: shardsConfig.main.binding };

/** Every shard, main first, then the split ranks in order. */
export const ALL_SHARDS: Shard[] = [
  MAIN_SHARD,
  ...Object.entries(SPLIT).flatMap(([n, cfg]) =>
    cfg.databases.map((d, i) => ({ key: `n${n}.${i}`, binding: d.binding, rank: Number(n), bucket: i }))),
];

export function isSplitRank(n: number): boolean {
  return Object.hasOwn(SPLIT, String(n));
}

/** Bucket of an id within a split rank: first hex digit of the hash mod buckets. */
export function bucketOf(id: string, n: number): number {
  const hash = id.slice(id.lastIndexOf(".") + 1);
  return parseInt(hash[0] ?? "0", 16) % SPLIT[String(n)]!.buckets;
}

/** Shards that can hold rows of rank `n` (all shards when n is unknown). */
export function shardsForRank(n: number | undefined): Shard[] {
  if (n === undefined) return ALL_SHARDS;
  if (!isSplitRank(n)) return [MAIN_SHARD];
  return ALL_SHARDS.filter((s) => s.rank === n);
}

export function shardForId(id: string): Shard | null {
  const n = rankFromId(id);
  if (n === null) return null;
  if (!isSplitRank(n)) return MAIN_SHARD;
  const b = bucketOf(id, n);
  return ALL_SHARDS.find((s) => s.rank === n && s.bucket === b) ?? null;
}

function bindingOf(env: Env, shard: Shard): D1Database {
  const db = (env as unknown as Record<string, D1Database>)[shard.binding];
  if (!db) throw new Error(`D1 binding ${shard.binding} is not configured`);
  return db;
}

export function dbOf(env: Env, shard: Shard): Database {
  return drizzle(bindingOf(env, shard), { schema });
}

/** The main database (global tables; ranks that are not split). */
export function mainDb(env: Env): Database {
  return dbOf(env, MAIN_SHARD);
}

/**
 * Rank encoded in a `Q.n{k}.{hash}` / `MC.n{k}.{hash}` id, or null if the id
 * is malformed.
 */
export function rankFromId(id: string): number | null {
  const m = /^(?:Q|MC)\.n(\d+)\.[0-9a-f]{16}$/.exec(id);
  return m ? Number(m[1]) : null;
}

/** Drizzle handle for the shard holding `id`, or null for a malformed id. */
export function dbForId(env: Env, id: string): Database | null {
  const s = shardForId(id);
  return s ? dbOf(env, s) : null;
}

/** Compatibility: the database for rank `n` when the rank is not split. */
export function dbFor(env: Env, n: number): Database {
  const shards = shardsForRank(n);
  if (shards.length !== 1) throw new Error(`rank ${n} is split across shards; use shardsForRank`);
  return dbOf(env, shards[0]!);
}
