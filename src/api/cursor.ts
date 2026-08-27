/**
 * Opaque keyset cursors.
 *
 * A cursor is the base64url of a JSON array of key values in sort order,
 * prefixed with a one-letter version. Clients must treat it as opaque and
 * pass it back verbatim as ?cursor=. Every list endpoint returns
 * `next_cursor` (null when the listing is exhausted).
 */

import { and, asc, desc, gt, isNotNull, isNull, lt, or, sql, type SQL } from "drizzle-orm";
import type { SQLiteColumn } from "drizzle-orm/sqlite-core";
import { BadRequest } from "./errors";

const VERSION = "k";

export type Key = (string | number | null)[];

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
 * Keyset predicate "row comes strictly after `key`" for an ORDER BY of
 * `columns` with matching `dirs`, where the last column is a unique tiebreak.
 * NULLs follow SQLite's ordering (first in ASC, last in DESC).
 */
export function afterKey(
  columns: SQLiteColumn[], dirs: ("asc" | "desc")[], key: Key,
): SQL {
  // Lexicographic: OR over prefixes where all earlier columns are equal
  // (NULL-safe) and this column is strictly "after".
  const branches: SQL[] = [];
  for (let i = 0; i < columns.length; i++) {
    const eqs: SQL[] = [];
    for (let j = 0; j < i; j++) eqs.push(nullSafeEq(columns[j]!, key[j]!));
    const strict = strictlyAfter(columns[i]!, dirs[i]!, key[i]!);
    branches.push(eqs.length ? and(...eqs, strict)! : strict);
  }
  return or(...branches)!;
}

function nullSafeEq(col: SQLiteColumn, v: string | number | null): SQL {
  return v === null ? isNull(col) : sql`${col} = ${v}`;
}

function strictlyAfter(col: SQLiteColumn, dir: "asc" | "desc", v: string | number | null): SQL {
  if (dir === "asc") {
    // NULL < everything. After NULL: any non-null. After v: col > v.
    return v === null ? isNotNull(col) : gt(col, v);
  }
  // DESC: everything > NULL comes first, NULLs last. After v: col < v OR col IS NULL.
  return v === null ? sql`0` : or(lt(col, v), isNull(col))!;
}

export function orderBy(columns: SQLiteColumn[], dirs: ("asc" | "desc")[]): SQL[] {
  return columns.map((c, i) => (dirs[i] === "desc" ? desc(c) : asc(c)));
}
