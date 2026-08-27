/**
 * Bulk export of any filtered cut, streamed from keyset-paged reads.
 *
 *   GET /export        CSV (UTF-8 BOM, CRLF, TRUE/FALSE, empty cell for null;
 *                      column order = EXPORT_COLUMNS, stable since v1)
 *   GET /export.csv    same
 *   GET /export.ndjson one JSON object per line, resumable: pass the value of
 *                      the X-Next-Cursor response header back as ?cursor=
 *                      (limit <= 5000 rows per response; omit for streaming
 *                      the whole cut in one response)
 *
 * Excel is generated client-side from CSV; format=xlsx is rejected by design.
 * Each export start is logged to the downloads table (best-effort).
 */

import { and, eq } from "drizzle-orm";
import { Hono, type Context } from "hono";
import {
  classNicknames as nick,
  downloads,
  labelings as lab,
  mutationClasses as mc,
  quivers as q,
} from "../db/schema";
import { dbFor, type Database } from "../db/shard";
import { afterKey, decodeCursor, encodeCursor, orderBy, type Key } from "./cursor";
import { parseInteger } from "./errors";
import { filterConditions, filtersAsRecord, parseFilters, type ListFilters } from "./quivers";

const PAGE = 500;

export const EXPORT_COLUMNS = [
  // --- quiver (per unlabeled quiver / per labeling) ---
  "qmd_id", "num_vertices", "exchange_matrix", "representation_type",
  "max_edge", "is_acyclic", "is_connected", "is_bipartite", "is_abundant",
  "is_planar", "symmetry_order", "symmetry_name",
  // --- mutation-class statistics ---
  "mc_id", "dynkin_type", "is_open", "class_size", "labeled_size",
  "distinct_quiver_count", "merged_orbit_count",
  "is_finite_confirmed", "is_infinite_confirmed", "is_infinite_expected",
  "size_of_explored_frontier", "is_mutation_acyclic",
  "is_banff", "is_louise", "is_p_prime",
  // --- phase 2 (appended so old column positions are unchanged) ---
  "exploration", "nickname",
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
  id: q.id, n: q.n, exchangeMatrix: q.exchangeMatrix,
  representationType: q.representationType, maxEdge: q.maxEdge,
  isAcyclic: q.isAcyclic, isConnected: q.isConnected,
  isBipartite: q.isBipartite, isAbundant: q.isAbundant, isPlanar: q.isPlanar,
  symmetryGroup: q.symmetryGroup, mcId: q.mutationClassId,
  mcDynkinType: mc.dynkinType, mcIsOpen: mc.isOpen, mcExploration: mc.exploration,
  mcClassSize: mc.classSize,
  mcDistinct: mc.distinctQuiverCount, mcMerged: mc.mergedOrbitCount,
  mcFinite: mc.isFiniteConfirmed, mcInfinite: mc.isInfiniteConfirmed,
  mcInfExpected: mc.isInfiniteExpected, mcFrontier: mc.sizeOfExploredFrontier,
  mcMutAcyclic: mc.isMutationAcyclic, mcBanff: mc.isBanff,
  mcLouise: mc.isLouise, mcPPrime: mc.isPPrime, nickname: nick.nickname,
};

type ExportRow = Awaited<ReturnType<typeof fetchDistinctPage>>[number];

/** Distinct quivers in (n, id) order after `after` (keyset). */
function fetchDistinctPage(db: Database, filters: ListFilters, after: Key | undefined, limit: number) {
  const cols = [q.n, q.id];
  const dirs: ("asc" | "desc")[] = ["asc", "asc"];
  const conds = filterConditions(filters);
  if (after) conds.push(afterKey(cols, dirs, after));
  return db.select(EXPORT_SELECTION)
    .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id)).leftJoin(nick, eq(nick.mcId, mc.id))
    .where(conds.length ? and(...conds) : undefined)
    .orderBy(...orderBy(cols, dirs)).limit(limit);
}

/** Labelings in (n, id, ord) order after `after` (keyset). */
function fetchLabelingsPage(db: Database, filters: ListFilters, after: Key | undefined, limit: number) {
  const cols = [q.n, q.id, lab.ord];
  const dirs: ("asc" | "desc")[] = ["asc", "asc", "asc"];
  const conds = filterConditions(filters);
  if (after) conds.push(afterKey(cols, dirs, after));
  return db.select({ ...EXPORT_SELECTION, ord: lab.ord, labMatrix: lab.matrix })
    .from(lab).innerJoin(q, eq(q.id, lab.qmdId))
    .leftJoin(mc, eq(q.mutationClassId, mc.id)).leftJoin(nick, eq(nick.mcId, mc.id))
    .where(and(...conds)).orderBy(...orderBy(cols, dirs)).limit(limit);
}

/** Flat export dict for one quiver + its class statistics (legacy _export_row). */
export function exportRow(r: ExportRow, matrix?: unknown): Record<string, unknown> {
  return {
    qmd_id: r.id,
    num_vertices: r.n,
    exchange_matrix: JSON.stringify(matrix ?? r.exchangeMatrix),
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
    exploration: r.mcExploration,
    nickname: r.nickname,
  };
}

/**
 * Iterate the rows of a cut in export order, keyset-paged. Yields
 * [row, keyAfterThisRow] so callers can emit a resumable cursor.
 */
async function* iterateRows(db: Database, filters: ListFilters, scope: string,
                            start: Key | undefined, max: number | null) {
  let after = start;
  let emitted = 0;
  for (;;) {
    const want = max === null ? PAGE : Math.min(PAGE, max - emitted);
    if (want <= 0) return;
    if (scope === "labelings") {
      const page = await fetchLabelingsPage(db, filters, after, want);
      for (const r of page) {
        after = [r.n, r.id, r.ord];
        emitted += 1;
        yield [exportRow(r, r.labMatrix), after] as const;
      }
      if (page.length < want) return;
    } else {
      const page = await fetchDistinctPage(db, filters, after, want);
      for (const r of page) {
        after = [r.n, r.id];
        emitted += 1;
        yield [exportRow(r), after] as const;
      }
      if (page.length < want) return;
    }
  }
}

export const exportRoutes = new Hono<{ Bindings: Env }>();

function logDownload(c: Context<{ Bindings: Env }>, db: Database, fmt: string,
                     loggedFilters: Record<string, unknown>, rowCount: () => number) {
  const email = c.req.query("email") || null;
  const name = c.req.query("name") || null;
  const ip = c.req.header("cf-connecting-ip")
    ?? c.req.header("x-forwarded-for")?.split(",")[0]?.trim() ?? null;
  return async () => {
    try {
      await db.insert(downloads).values({
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

async function handleCsv(c: Context<{ Bindings: Env }>) {
  const fmt = (c.req.query("format") ?? "csv").toLowerCase();
  if (fmt !== "csv") {
    return c.json({ detail: "format must be 'csv' (Excel files are generated client-side from CSV)" }, 400);
  }
  const scope = scopeOf(c);
  if (!scope) return c.json({ detail: "scope must be 'distinct' or 'labelings'" }, 400);
  const filters = parseFilters((k) => c.req.query(k));
  const db = dbFor(c.env, filters.rank ?? 0);
  const loggedFilters = { scope, ...filtersAsRecord(filters) };

  const { readable, writable } = new TransformStream<Uint8Array>();
  const encoder = new TextEncoder();
  let rowCount = 0;
  const finishLog = logDownload(c, db, "csv", loggedFilters, () => rowCount);

  const pump = async () => {
    const writer = writable.getWriter();
    try {
      // UTF-8 BOM so Excel opens unicode cleanly on double-click.
      await writer.write(encoder.encode("\uFEFF" + EXPORT_COLUMNS.join(",") + "\r\n"));
      let chunk = "";
      for await (const [row] of iterateRows(db, filters, scope, undefined, null)) {
        chunk += csvLine(row);
        rowCount += 1;
        if (chunk.length > 64_000) { await writer.write(encoder.encode(chunk)); chunk = ""; }
      }
      if (chunk) await writer.write(encoder.encode(chunk));
      await writer.close();
    } catch (err) {
      await writer.abort(err);
      throw err;
    } finally {
      await finishLog();
    }
  };
  c.executionCtx.waitUntil(pump());

  const base = scope === "labelings" ? "qmd-labelings" : "qmd-quivers";
  return new Response(readable, {
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
  const db = dbFor(c.env, filters.rank ?? 0);
  const limitParam = parseInteger("limit", c.req.query("limit"));
  const max = limitParam === undefined ? null : Math.min(Math.max(limitParam, 1), 5000);
  const start = decodeCursor(c.req.query("cursor"), scope === "labelings" ? 3 : 2);

  // Bounded page: buffer (<= 5000 rows) so the resume cursor can be a header.
  if (max !== null) {
    let lastKey: Key | undefined;
    let count = 0;
    let body = "";
    for await (const [row, key] of iterateRows(db, filters, scope, start, max)) {
      body += JSON.stringify(row) + "\n";
      lastKey = key;
      count += 1;
    }
    const next = count === max && lastKey ? encodeCursor(lastKey) : "";
    if (!start) c.executionCtx.waitUntil(logDownload(c, db, "ndjson", { scope, ...filtersAsRecord(filters) }, () => count)());
    return new Response(body, {
      headers: {
        "Content-Type": "application/x-ndjson; charset=utf-8",
        "X-Next-Cursor": next,
        "X-Row-Count": String(count),
        "Cache-Control": "no-store",
      },
    });
  }

  // Unbounded: stream the whole cut.
  const { readable, writable } = new TransformStream<Uint8Array>();
  const encoder = new TextEncoder();
  let count = 0;
  const finishLog = logDownload(c, db, "ndjson", { scope, ...filtersAsRecord(filters) }, () => count);
  const pump = async () => {
    const writer = writable.getWriter();
    try {
      let chunk = "";
      for await (const [row] of iterateRows(db, filters, scope, start, null)) {
        chunk += JSON.stringify(row) + "\n";
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
  return new Response(readable, {
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
