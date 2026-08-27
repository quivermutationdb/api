/** Thrown for bad query params; the API router turns it into a 400 {detail}. */
export class BadRequest extends Error {}

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

export function parseDir(v: string | undefined): "asc" | "desc" {
  if (v === undefined || v === "asc") return "asc";
  if (v === "desc") return "desc";
  throw new BadRequest("dir must be 'asc' or 'desc'");
}
