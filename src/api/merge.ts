/**
 * Merge keyset pages from several shards into one ordered page.
 *
 * Each shard is asked for `limit + 1` rows after its own position; the pages
 * are merged by sort key; the composite cursor records, per shard, the key of
 * the last row emitted from it (or null when that shard is exhausted).
 */

import { compareKeys, decodeCursor, encodeCursor, type Dir, type Key } from "./cursor";
import { BadRequest } from "./errors";

export interface ShardPage<R> { shardKey: string; rows: R[] }

export interface MergeInput<R> {
  shardKeys: string[];
  dirs: Dir[];
  keyOf: (row: R) => Key;
  /** Fetch up to `limit` rows from a shard strictly after `after` (undefined = from the start). */
  fetch: (shardKey: string, after: Key | undefined, limit: number) => Promise<R[]>;
  limit: number;
  /** Offset for page-number UIs (only honoured on the first page, no cursor). */
  offset?: number;
  cursor?: string;
}

/**
 * Composite cursor: JSON object {shardKey: Key | null | "start"} inside the
 * opaque wrapper — the last key emitted from that shard, null when the shard
 * is exhausted, "start" when nothing has been taken from it yet.
 */
type Position = Key | null | "start";

function decodeComposite(raw: string | undefined, shardKeys: string[], arity: number): Record<string, Position> | undefined {
  if (!raw) return undefined;
  const k = decodeCursor(raw, 1);
  if (!k || typeof k[0] !== "string") throw new BadRequest("invalid cursor");
  let obj: Record<string, Position>;
  try { obj = JSON.parse(k[0] as string); } catch { throw new BadRequest("invalid cursor"); }
  for (const s of shardKeys) {
    const v = obj[s];
    if (v === undefined) throw new BadRequest("invalid cursor");
    if (v !== null && v !== "start" && (!Array.isArray(v) || v.length !== arity)) throw new BadRequest("invalid cursor");
  }
  return obj;
}

function afterOf(pos: Position | undefined): Key | undefined {
  return pos === undefined || pos === null || pos === "start" ? undefined : pos;
}

export async function mergeShards<R>(m: MergeInput<R>): Promise<{ items: R[]; next_cursor: string | null }> {
  const single = m.shardKeys.length === 1;
  const arity = m.dirs.length;
  const positions = decodeComposite(m.cursor, m.shardKeys, arity);
  const offset = positions ? 0 : (m.offset ?? 0);

  // Single shard: plain keyset (offset applies directly).
  if (single) {
    const sk = m.shardKeys[0]!;
    if (positions && positions[sk] === null) return { items: [], next_cursor: null };
    const after = afterOf(positions?.[sk]);
    const rows = await m.fetch(sk, after, m.limit + 1 + offset);
    const page = rows.slice(offset, offset + m.limit);
    const more = rows.length > offset + m.limit;
    const last = page[page.length - 1];
    return { items: page, next_cursor: more && last ? encodeCursor([JSON.stringify({ [sk]: m.keyOf(last) })]) : null };
  }

  // Multi-shard: fetch limit+offset+1 from every live shard, merge.
  const want = m.limit + offset + 1;
  const pages = await Promise.all(m.shardKeys.map(async (sk) => {
    if (positions && positions[sk] === null) return { shardKey: sk, rows: [] as R[] };
    return { shardKey: sk, rows: await m.fetch(sk, afterOf(positions?.[sk]), want) };
  }));
  const idx = new Map(pages.map((p) => [p.shardKey, 0]));
  const merged: { row: R; shardKey: string }[] = [];
  while (merged.length < want) {
    let best: { row: R; shardKey: string } | null = null;
    for (const p of pages) {
      const i = idx.get(p.shardKey)!;
      const row = p.rows[i];
      if (row === undefined) continue;
      if (!best || compareKeys(m.keyOf(row), m.keyOf(best.row), m.dirs) < 0) best = { row, shardKey: p.shardKey };
    }
    if (!best) break;
    merged.push(best);
    idx.set(best.shardKey, idx.get(best.shardKey)! + 1);
  }
  const page = merged.slice(offset, offset + m.limit);
  const more = merged.length > offset + m.limit;
  if (!more) return { items: page.map((x) => x.row), next_cursor: null };
  // Positions: last emitted key per shard; a shard whose page was shorter than
  // `want` and fully consumed is exhausted (null).
  const next: Record<string, Position> = {};
  for (const p of pages) {
    const consumed = merged.slice(0, offset + m.limit).filter((x) => x.shardKey === p.shardKey);
    const lastRow = consumed[consumed.length - 1];
    if (lastRow) next[p.shardKey] = m.keyOf(lastRow.row);
    else if (positions && positions[p.shardKey] === null) next[p.shardKey] = null;
    else if (p.rows.length === 0) next[p.shardKey] = null;            // fetched nothing: exhausted
    else next[p.shardKey] = positions?.[p.shardKey] ?? "start";       // untouched this page
  }
  return { items: page.map((x) => x.row), next_cursor: encodeCursor([JSON.stringify(next)]) };
}
