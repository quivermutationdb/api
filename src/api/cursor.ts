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
import type { SQLiteColumn } from "drizzle-orm/sqlite-core";
import { BadRequest } from "./errors";

const VERSION = "k";

export type Key = (string | number | null)[];
export type Dir = "asc" | "desc";
export type KeyCol = SQLiteColumn | SQL;

export function encodeCursor(key: Key): string {
  const json = JSON.stringify(key);
  const b64 = btoa(unescape(encodeURIComponent(json)));
  return VERSION + b64.replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

export function decodeCursor(raw: string | undefined, arity: number): Key | undefined {
  if (raw === undefined || raw === "") return undefined;
  if (!raw.startsWith(VERSION)) throw new BadRequest("invalid cursor");
  try {
    const b64 = raw.slice(1).replaceAll("-", "+").replaceAll("_", "/");
    const key = JSON.parse(decodeURIComponent(escape(atob(b64))));
    if (!Array.isArray(key) || key.length !== arity
        || !key.every((v) => v === null || typeof v === "string" || typeof v === "number")) {
      throw new Error();
    }
    return key as Key;
  } catch {
    throw new BadRequest("invalid cursor");
  }
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

export function orderBy(columns: KeyCol[], dirs: Dir[]): SQL[] {
  return columns.map((c, i) => (dirs[i] === "desc" ? sql`${c} desc` : sql`${c} asc`));
}

/** Compare two keys under `dirs` (SQLite NULL ordering); used to merge shard pages. */
export function compareKeys(a: Key, b: Key, dirs: Dir[]): number {
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? null, y = b[i] ?? null;
    if (x === y) continue;
    const dir = dirs[i] === "desc" ? -1 : 1;
    if (x === null) return -1 * dir;          // NULL first in ASC, last in DESC
    if (y === null) return 1 * dir;
    return (x < y ? -1 : 1) * dir;
  }
  return 0;
}
