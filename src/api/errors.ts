/** Thrown for bad query params; the API router turns it into a 400 {detail}. */
export class BadRequest extends Error {}

/**
 * Thrown when the database is reachable but cannot serve the request right now
 * -- a connection-pool timeout, or the instance refusing new connections. The
 * router turns it into a 503 with Retry-After.
 *
 * D1 has no connection pool, so nothing raises this today. It exists because a
 * pooled Postgres behind Hyperdrive does, and without it pool exhaustion
 * surfaces as an untyped 500 that tells a client (and an agent following
 * /llms.txt) to give up rather than retry.
 */
export class Unavailable extends Error {
  constructor(message = "Database temporarily unavailable", readonly retryAfter = 2) {
    super(message);
  }
}

/**
 * Postgres SQLSTATE 57014, query_canceled -- here, always the server-side
 * statement_timeout. It means the query as written cannot be served at this
 * scale, so it is NOT a 503: retrying an identical request will time out
 * identically, and telling an agent to retry would just burn its budget.
 */
export const QUERY_CANCELED = "57014";

/**
 * Walks the cause chain: Drizzle wraps driver errors in a DrizzleQueryError
 * carrying the query text, so the pg error -- and its SQLSTATE -- is at
 * `.cause`, not the top level. Checking only the top level silently misses
 * every timeout and reports it as a 500.
 */
export function isStatementTimeout(e: unknown): boolean {
  for (let cur: unknown = e, depth = 0; cur && depth < 5; depth++) {
    if (typeof cur !== "object") break;
    if ((cur as { code?: unknown }).code === QUERY_CANCELED) return true;
    cur = (cur as { cause?: unknown }).cause;
  }
  return false;
}

export function parseBool(name: string, v: string | undefined): boolean | undefined {
  if (v === undefined || v === "") return undefined;
  const s = v.toLowerCase();
  if (s === "true" || s === "1") return true;
  if (s === "false" || s === "0") return false;
  throw new BadRequest(`${name} must be true or false`);
}

export function parseInteger(name: string, v: string | undefined): number | undefined {
  if (v === undefined || v === "") return undefined;
  if (!/^-?\d{1,12}$/.test(v)) throw new BadRequest(`${name} must be an integer`);
  return Number(v);
}

/** offset/limit with the same clamping everywhere. */
export function parsePaging(get: (k: string) => string | undefined,
                            defaultLimit: number, maxLimit = 1000) {
  return {
    offset: Math.max(parseInteger("offset", get("offset")) ?? 0, 0),
    limit: Math.min(Math.max(parseInteger("limit", get("limit")) ?? defaultLimit, 1), maxLimit),
  };
}

export type TotalMode = "capped" | "exact";

/**
 * ?total= -- how hard the server should work for the result count.
 * "capped" (default) stops counting past TOTAL_CAP and marks the answer a
 * lower bound; "exact" pays for a full count. See totalsFor in quivers.ts.
 */
export function parseTotalMode(v: string | undefined): TotalMode {
  if (v === undefined || v === "capped") return "capped";
  if (v === "exact") return "exact";
  throw new BadRequest("total must be 'capped' or 'exact'");
}

export function parseDir(v: string | undefined): "asc" | "desc" {
  if (v === undefined || v === "asc") return "asc";
  if (v === "desc") return "desc";
  throw new BadRequest("dir must be 'asc' or 'desc'");
}
