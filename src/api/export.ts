/**
 * Bulk export of any filtered cut, streamed from keyset-paged reads.
 *
 *   GET /export        CSV (UTF-8 BOM, CRLF, TRUE/FALSE, empty cell for null;
 *                      column order = EXPORT_COLUMNS, stable since v1)
 *   GET /export.csv    same
 *   GET /export.ndjson one JSON object per line, resumable: pass the value of
 *                      the X-Next-Cursor response header back as ?cursor=
 *                      (limit <= 5000 rows per response; omit to stream all)
 *
 * Order: rank ascending, then shard, then id (rowid) — resumable but not a
 * global id order across the shards of a split rank.
 */

import { and, eq, sql } from "drizzle-orm";
import { Hono, type Context } from "hono";
import { decodeUpper } from "../db/matrix";
import { classNicknames as nick, downloads, labelings as lab, mutationClasses as mc, quivers as q } from "../db/schema";
import { ALL_SHARDS, dbOf, mainDb, shardsForRank, type Database, type Shard } from "../db/shard";
import { afterKey, decodeCursor, encodeCursor, orderBy, type Dir, type Key, type KeyCol } from "./cursor";
import { BadRequest, parseInteger } from "./errors";
import { filterConditions, filtersAsRecord, parseFilters, type ListFilters } from "./quivers";

const PAGE = 500;
const Q_ROWID = sql`${q}.rowid`;

export const EXPORT_COLUMNS = [
  "qmd_id", "num_vertices", "exchange_matrix", "representation_type",
  "max_edge", "is_acyclic", "is_connected", "is_bipartite", "is_abundant",
  "is_planar", "symmetry_order", "symmetry_name",
  "mc_id", "dynkin_type", "is_open", "class_size", "labeled_size",
  "distinct_quiver_count", "merged_orbit_count",
  "is_finite_confirmed", "is_infinite_confirmed", "is_infinite_expected",
  "size_of_explored_frontier", "is_mutation_acyclic",
  "is_banff", "is_louise", "is_p_prime",
  "exploration", "nickname",
  // --- phase 3 (appended) ---
  "mutation_finite", "explored",
] as const;

function cell(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "boolean") return v ? "TRUE" : "FALSE";
  const s = String(v);
  return /[",\r\n]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s;
}

function csvLine(row: Record<string, unknown>): string {
  return EXPORT_COLUMNS.map((c) => cell(row[c])).join(",") + "\r\n";
}

const EXPORT_SELECTION = {
  rowid: sql<number>`${q}.rowid`,
  id: q.id, n: q.n, exchangeMatrix: q.exchangeMatrix,
  representationType: q.representationType, maxEdge: q.maxEdge,
  isAcyclic: q.isAcyclic, isConnected: q.isConnected,
  isBipartite: q.isBipartite, isAbundant: q.isAbundant, isPlanar: q.isPlanar,
  symmetryGroup: q.symmetryGroup, mcId: q.mutationClassId, mutationFinite: q.mutationFinite,
  mcDynkinType: mc.dynkinType, mcIsOpen: mc.isOpen, mcExploration: mc.exploration,
  mcClassSize: mc.classSize,
  mcDistinct: mc.distinctQuiverCount, mcMerged: mc.mergedOrbitCount,
  mcFinite: mc.isFiniteConfirmed, mcInfinite: mc.isInfiniteConfirmed,
  mcInfExpected: mc.isInfiniteExpected, mcFrontier: mc.sizeOfExploredFrontier,
  mcMutAcyclic: mc.isMutationAcyclic, mcBanff: mc.isBanff,
  mcLouise: mc.isLouise, mcPPrime: mc.isPPrime, nickname: nick.nickname,
};

type ExportRow = Awaited<ReturnType<typeof fetchDistinctPage>>[number];

function fetchDistinctPage(db: Database, filters: ListFilters, after: Key | undefined, limit: number) {
  const cols: KeyCol[] = [q.n, Q_ROWID];
  const dirs: Dir[] = ["asc", "asc"];
  const conds = filterConditions(filters);
  if (after) conds.push(afterKey(cols, dirs, after));
  return db.select(EXPORT_SELECTION)
    .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id)).leftJoin(nick, eq(nick.mcId, mc.id))
    .where(conds.length ? and(...conds) : undefined)
    .orderBy(...orderBy(cols, dirs)).limit(limit);
}

function fetchLabelingsPage(db: Database, filters: ListFilters, after: Key | undefined, limit: number) {
  const cols: KeyCol[] = [q.n, Q_ROWID, lab.ord];
  const dirs: Dir[] = ["asc", "asc", "asc"];
  const conds = filterConditions(filters);
  if (after) conds.push(afterKey(cols, dirs, after));
  return db.select({ ...EXPORT_SELECTION, ord: lab.ord, labMatrix: lab.matrix })
    .from(lab).innerJoin(q, eq(q.id, lab.qmdId))
    .leftJoin(mc, eq(q.mutationClassId, mc.id)).leftJoin(nick, eq(nick.mcId, mc.id))
    .where(and(...conds)).orderBy(...orderBy(cols, dirs)).limit(limit);
}

/** Flat export dict for one quiver + its class statistics (legacy _export_row + phase 2/3 fields). */
export function exportRow(r: ExportRow, matrix?: string): Record<string, unknown> {
  return {
    qmd_id: r.id,
    num_vertices: r.n,
    exchange_matrix: JSON.stringify(decodeUpper(r.n, matrix ?? r.exchangeMatrix)),
    representation_type: r.representationType,
    max_edge: r.maxEdge,
    is_acyclic: r.isAcyclic,
    is_connected: r.isConnected,
    is_bipartite: r.isBipartite,
    is_abundant: r.isAbundant,
    is_planar: r.isPlanar,
    symmetry_order: r.symmetryGroup?.order ?? null,
    symmetry_name: r.symmetryGroup?.name ?? null,
    mc_id: r.mcId,
    dynkin_type: r.mcDynkinType,
    is_open: r.mcIsOpen ?? false,
    class_size: r.mcIsOpen === false ? r.mcClassSize : null,
    labeled_size: r.mcClassSize,
    distinct_quiver_count: r.mcDistinct,
    merged_orbit_count: r.mcMerged,
    is_finite_confirmed: r.mcFinite,
    is_infinite_confirmed: r.mcInfinite,
    is_infinite_expected: r.mcInfExpected,
    size_of_explored_frontier: r.mcFrontier,
    is_mutation_acyclic: r.mcMutAcyclic,
    is_banff: r.mcBanff,
    is_louise: r.mcLouise,
    is_p_prime: r.mcPPrime,
    exploration: r.mcId ? r.mcExploration : null,
    nickname: r.nickname,
    mutation_finite: r.mutationFinite,
    explored: r.mcId !== null,
  };
}

/**
 * Walk a cut in export order across shards. The resume key is
 * [shardIndex, n, rowid(, ord)]: shard index into the ordered shard list for
 * the cut, then the per-shard keyset key.
 */
async function* iterateRows(env: Env, filters: ListFilters, scope: string,
                            start: Key | undefined, max: number | null) {
  const shards: Shard[] = filters.rank !== undefined ? shardsForRank(filters.rank) : ALL_SHARDS;
  let si = start ? (start[0] as number) : 0;
  let after: Key | undefined = start ? start.slice(1) : undefined;
  let emitted = 0;
  for (; si < shards.length; si++, after = undefined) {
    const db = dbOf(env, shards[si]!);
    for (;;) {
      const want = max === null ? PAGE : Math.min(PAGE, max - emitted);
      if (want <= 0) return;
      if (scope === "labelings") {
        const page = await fetchLabelingsPage(db, filters, after, want);
        for (const r of page) {
          after = [r.n, r.rowid, r.ord];
          emitted += 1;
          yield [exportRow(r, r.labMatrix), [si, ...after] as Key] as const;
        }
        if (page.length < want) break;
      } else {
        const page = await fetchDistinctPage(db, filters, after, want);
        for (const r of page) {
          after = [r.n, r.rowid];
          emitted += 1;
          yield [exportRow(r), [si, ...after] as Key] as const;
        }
        if (page.length < want) break;
      }
    }
  }
}

export const exportRoutes = new Hono<{ Bindings: Env }>();

function logDownload(c: Context<{ Bindings: Env }>, fmt: string,
                     loggedFilters: Record<string, unknown>, rowCount: () => number) {
  const email = c.req.query("email") || null;
  const name = c.req.query("name") || null;
  const ip = c.req.header("cf-connecting-ip")
    ?? c.req.header("x-forwarded-for")?.split(",")[0]?.trim() ?? null;
  return async () => {
    try {
      await mainDb(c.env).insert(downloads).values({
        createdAt: new Date().toISOString().replace("T", " ").slice(0, 19),
        fmt, rowCount: rowCount(), filters: loggedFilters,
        email: email?.slice(0, 254) ?? null, name: name?.slice(0, 254) ?? null,
        ip, userAgent: c.req.header("user-agent") ?? null, referer: c.req.header("referer") ?? null,
      });
    } catch (e) {
      console.error("download logging failed", e);
    }
  };
}

function scopeOf(c: Context<{ Bindings: Env }>): "distinct" | "labelings" | null {
  const scope = (c.req.query("scope") ?? "distinct").toLowerCase();
  return scope === "distinct" || scope === "labelings" ? scope : null;
}

function stamp(): string {
  const iso = new Date().toISOString();
  return iso.slice(0, 10).replaceAll("-", "") + "-" + iso.slice(11, 19).replaceAll(":", "");
}

function streamed(c: Context<{ Bindings: Env }>, filters: ListFilters, scope: string, start: Key | undefined,
                  fmt: "csv" | "ndjson", line: (row: Record<string, unknown>) => string, header: string) {
  const { readable, writable } = new TransformStream<Uint8Array>();
  const encoder = new TextEncoder();
  let count = 0;
  const finishLog = logDownload(c, fmt, { scope, ...filtersAsRecord(filters) }, () => count);
  const pump = async () => {
    const writer = writable.getWriter();
    try {
      if (header) await writer.write(encoder.encode(header));
      let chunk = "";
      for await (const [row] of iterateRows(c.env, filters, scope, start, null)) {
        chunk += line(row);
        count += 1;
        if (chunk.length > 64_000) { await writer.write(encoder.encode(chunk)); chunk = ""; }
      }
      if (chunk) await writer.write(encoder.encode(chunk));
      await writer.close();
    } catch (err) {
      await writer.abort(err);
      throw err;
    } finally {
      if (!start) await finishLog();
    }
  };
  c.executionCtx.waitUntil(pump());
  return readable;
}

async function handleCsv(c: Context<{ Bindings: Env }>) {
  const fmt = (c.req.query("format") ?? "csv").toLowerCase();
  if (fmt !== "csv") {
    return c.json({ detail: "format must be 'csv' (Excel files are generated client-side from CSV)" }, 400);
  }
  const scope = scopeOf(c);
  if (!scope) return c.json({ detail: "scope must be 'distinct' or 'labelings'" }, 400);
  const filters = parseFilters((k) => c.req.query(k));
  const body = streamed(c, filters, scope, undefined, "csv", csvLine, "\uFEFF" + EXPORT_COLUMNS.join(",") + "\r\n");
  const base = scope === "labelings" ? "qmd-labelings" : "qmd-quivers";
  return new Response(body, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="${base}-${stamp()}.csv"`,
      "Cache-Control": "no-store",
    },
  });
}

async function handleNdjson(c: Context<{ Bindings: Env }>) {
  const scope = scopeOf(c);
  if (!scope) return c.json({ detail: "scope must be 'distinct' or 'labelings'" }, 400);
  const filters = parseFilters((k) => c.req.query(k));
  const limitParam = parseInteger("limit", c.req.query("limit"));
  const max = limitParam === undefined ? null : Math.min(Math.max(limitParam, 1), 5000);
  const arity = scope === "labelings" ? 4 : 3;
  const start = decodeCursor(c.req.query("cursor"), arity);
  if (start && typeof start[0] !== "number") throw new BadRequest("invalid cursor");

  if (max !== null) {
    let lastKey: Key | undefined;
    let count = 0;
    let body = "";
    for await (const [row, key] of iterateRows(c.env, filters, scope, start, max)) {
      body += JSON.stringify(row) + "\n";
      lastKey = key;
      count += 1;
    }
    const next = count === max && lastKey ? encodeCursor(lastKey) : "";
    if (!start) c.executionCtx.waitUntil(logDownload(c, "ndjson", { scope, ...filtersAsRecord(filters) }, () => count)());
    return new Response(body, {
      headers: {
        "Content-Type": "application/x-ndjson; charset=utf-8",
        "X-Next-Cursor": next,
        "X-Row-Count": String(count),
        "Cache-Control": "no-store",
      },
    });
  }
  const body = streamed(c, filters, scope, start, "ndjson", (row) => JSON.stringify(row) + "\n", "");
  return new Response(body, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Content-Disposition": `attachment; filename="qmd-${scope}-${stamp()}.ndjson"`,
      "Cache-Control": "no-store",
    },
  });
}

exportRoutes.get("/export", handleCsv);
exportRoutes.get("/export.csv", handleCsv);
exportRoutes.get("/export.ndjson", handleNdjson);
