/**
 * Opaque keyset cursors.
 *
 * A cursor is the base64url of a JSON array of key values in sort order,
 * prefixed with a one-letter version. Clients treat it as opaque and pass it
 * back verbatim as ?cursor=. Every list endpoint returns `next_cursor` (null
 * when exhausted). The last key column is always a unique tiebreak — for the
 * row tables that is the SQLite rowid (rows are inserted in id order per
 * rank, so (n, rowid) is id order).
 */

import { and, isNotNull, isNull, or, sql, type SQL } from "drizzle-orm";
import type { PgColumn } from "drizzle-orm/pg-core";
import { BadRequest } from "./errors";

// Bumped "k" -> "p" at the D1 -> Postgres cutover. Every cursor shape changed:
// list cursors dropped the {shardKey: key} wrapper and export cursors dropped
// their leading shard index. Without a version bump an old 3-element export
// cursor ([shardIndex, n, rowid]) would be silently readable as a new
// 3-element labelings key ([n, seq, ord]) -- an arity check cannot tell them
// apart. The prefix can, for every shape at once.
const VERSION = "p";
const LEGACY_VERSION = "k";

export type Key = (string | number | null)[];
export type Dir = "asc" | "desc";
export type KeyCol = PgColumn | SQL;

export function encodeCursor(key: Key): string {
  const json = JSON.stringify(key);
  const b64 = btoa(unescape(encodeURIComponent(json)));
  return VERSION + b64.replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

export function decodeCursor(raw: string | undefined, arity: number): Key | undefined {
  if (raw === undefined || raw === "") return undefined;
  if (raw.startsWith(LEGACY_VERSION)) {
    throw new BadRequest(
      "this cursor was issued by the pre-Postgres API and is no longer valid; "
      + "restart the walk without ?cursor= (the page order is unchanged)");
  }
  if (!raw.startsWith(VERSION)) throw new BadRequest("invalid cursor");
  let key: Key;
  try {
    const b64 = raw.slice(1).replaceAll("-", "+").replaceAll("_", "/");
    const parsed = JSON.parse(decodeURIComponent(escape(atob(b64))));
    if (!Array.isArray(parsed)
        || !parsed.every((v) => v === null || typeof v === "string" || typeof v === "number")) {
      throw new Error();
    }
    key = parsed as Key;
  } catch {
    throw new BadRequest("invalid cursor");
  }
  if (key.length !== arity) throw new BadRequest("invalid cursor");
  return key;
}

/**
 * Keyset predicate "row comes strictly after `key`" for ORDER BY `columns`
 * with `dirs`. NULLs follow SQLite's ordering (first in ASC, last in DESC).
 */
export function afterKey(columns: KeyCol[], dirs: Dir[], key: Key): SQL {
  const branches: SQL[] = [];
  for (let i = 0; i < columns.length; i++) {
    const eqs: SQL[] = [];
    for (let j = 0; j < i; j++) eqs.push(nullSafeEq(columns[j]!, key[j]!));
    const strict = strictlyAfter(columns[i]!, dirs[i]!, key[i]!);
    branches.push(eqs.length ? and(...eqs, strict)! : strict);
  }
  return or(...branches)!;
}

function nullSafeEq(col: KeyCol, v: string | number | null): SQL {
  return v === null ? isNull(col) : sql`${col} = ${v}`;
}

function strictlyAfter(col: KeyCol, dir: Dir, v: string | number | null): SQL {
  if (dir === "asc") return v === null ? isNotNull(col) : sql`${col} > ${v}`;
  return v === null ? sql`0` : or(sql`${col} < ${v}`, isNull(col))!;
}

/**
 * NULL placement is stated explicitly and must stay that way. Postgres's own
 * default is the opposite of what `afterKey` encodes -- NULLs last ascending,
 * first descending -- so dropping these clauses would silently reorder every
 * page over a nullable column (class_size, dynkin_type, mutation_finite) out of
 * agreement with the keyset predicate, which is how a paged walk starts
 * skipping rows. The chosen placement is SQLite's, kept deliberately so the
 * published page order did not change at the D1 -> Postgres cutover.
 */
export function orderBy(columns: KeyCol[], dirs: Dir[]): SQL[] {
  return columns.map((c, i) =>
    (dirs[i] === "desc" ? sql`${c} desc nulls last` : sql`${c} asc nulls first`));
}

/**
 * One keyset page from the single database. This is what is left of
 * src/api/merge.ts after the shards collapsed, and it keeps that module's
 * paging contract exactly: `offset` is honoured only on the first page (a
 * cursor supersedes it), one extra row is fetched to decide `next_cursor`, and
 * an exhausted walk returns null rather than an empty cursor.
 */
export async function keysetPage<R>(m: {
  dirs: Dir[];
  keyOf: (row: R) => Key;
  /** Rows strictly after `after` (undefined = from the start), in sort order. */
  fetch: (after: Key | undefined, limit: number) => Promise<R[]>;
  limit: number;
  offset?: number;
  cursor?: string;
}): Promise<{ items: R[]; next_cursor: string | null }> {
  const after = decodeCursor(m.cursor, m.dirs.length);
  const offset = after ? 0 : (m.offset ?? 0);
  const rows = await m.fetch(after, m.limit + 1 + offset);
  const page = rows.slice(offset, offset + m.limit);
  const more = rows.length > offset + m.limit;
  const last = page[page.length - 1];
  return { items: page, next_cursor: more && last ? encodeCursor(m.keyOf(last)) : null };
}
