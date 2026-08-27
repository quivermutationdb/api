/**
 * GET /export — CSV export of any filtered cut, streamed from paginated
 * reads, matching the Python exporter byte format (UTF-8 BOM, CRLF,
 * TRUE/FALSE booleans, empty cell for null) and column order
 * (the legacy backend's EXPORT_COLUMNS) so downloads stay diffable across
 * the migration.
 *
 * Excel is generated client-side from CSV (see the frontend's download.js);
 * format=xlsx is rejected here by design.
 *
 * Each export is logged to the downloads table (best-effort, after the
 * stream completes — a tracking failure must never break a download).
 */

import { and, eq, inArray } from "drizzle-orm";
import { Hono, type Context } from "hono";
import {
  downloads,
  mutationClasses as mc,
  mutationClassPayloads as payloads,
  quivers as q,
} from "../db/schema";
import { dbFor, type Database } from "../db/shard";
import { filterConditions, parseFilters, sortOrder, type ListFilters } from "./quivers";

const PAGE = 200;

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
  mcDynkinType: mc.dynkinType, mcIsOpen: mc.isOpen, mcClassSize: mc.classSize,
  mcDistinct: mc.distinctQuiverCount, mcMerged: mc.mergedOrbitCount,
  mcFinite: mc.isFiniteConfirmed, mcInfinite: mc.isInfiniteConfirmed,
  mcInfExpected: mc.isInfiniteExpected, mcFrontier: mc.sizeOfExploredFrontier,
  mcMutAcyclic: mc.isMutationAcyclic, mcBanff: mc.isBanff,
  mcLouise: mc.isLouise, mcPPrime: mc.isPPrime,
};

type ExportRow = Awaited<ReturnType<typeof fetchPage>>[number];

function fetchPage(db: Database, filters: ListFilters, offset: number) {
  const conds = filterConditions(filters);
  return db.select(EXPORT_SELECTION)
    .from(q).leftJoin(mc, eq(q.mutationClassId, mc.id))
    .where(conds.length ? and(...conds) : undefined)
    .orderBy(...sortOrder(undefined, undefined))
    .offset(offset).limit(PAGE);
}

/** Flat export dict for one quiver + its class statistics (legacy _export_row). */
function exportRow(r: ExportRow, matrix?: unknown): Record<string, unknown> {
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
  };
}

export const exportRoutes = new Hono<{ Bindings: Env }>();

async function handleExport(c: Context<{ Bindings: Env }>) {
  const get = (k: string) => c.req.query(k);
  const fmt = (get("format") ?? "csv").toLowerCase();
  if (fmt !== "csv") {
    return c.json({
      detail: "format must be 'csv' (Excel files are generated client-side from CSV)",
    }, 400);
  }
  const scope = (get("scope") ?? "distinct").toLowerCase();
  if (scope !== "distinct" && scope !== "labelings") {
    return c.json({ detail: "scope must be 'distinct' or 'labelings'" }, 400);
  }
  const filters = parseFilters(get);
  const db = dbFor(c.env, filters.rank ?? 0);

  const loggedFilters: Record<string, unknown> = { scope };
  for (const [k, v] of Object.entries({
    rank: filters.rank, dynkin_type: filters.dynkinType,
    representation_type: filters.representationType, max_edge: filters.maxEdge,
    is_open: filters.isOpen, orbit_min: filters.orbitMin,
    orbit_max: filters.orbitMax, is_acyclic: filters.isAcyclic,
    is_connected: filters.isConnected, is_simply_laced: filters.isSimplyLaced,
    is_mutation_finite: filters.isMutationFinite,
  })) {
    if (v !== undefined) loggedFilters[k] = v;
  }

  const email = get("email") || null;
  const name = get("name") || null;
  const ip = c.req.header("cf-connecting-ip")
    ?? c.req.header("x-forwarded-for")?.split(",")[0]?.trim() ?? null;
  const userAgent = c.req.header("user-agent") ?? null;
  const referer = c.req.header("referer") ?? null;

  const { readable, writable } = new TransformStream<Uint8Array>();
  const encoder = new TextEncoder();

  const pump = async () => {
    const writer = writable.getWriter();
    let rowCount = 0;
    try {
      // UTF-8 BOM so Excel opens unicode cleanly on double-click.
      await writer.write(encoder.encode("\uFEFF" + EXPORT_COLUMNS.join(",") + "\r\n"));
      for (let offset = 0; ; offset += PAGE) {
        const page = await fetchPage(db, filters, offset);
        if (page.length === 0) break;
        let chunk = "";
        if (scope === "labelings") {
          const mcIds = [...new Set(page.map((r) => r.mcId)
            .filter((x): x is string => x !== null))];
          const orbits = new Map<string, { qmd_id: string; matrix: unknown }[]>();
          for (let i = 0; i < mcIds.length; i += 50) {
            for (const b of await db
              .select({ id: payloads.mutationClassId, labeled: payloads.labeledQuivers })
              .from(payloads)
              .where(inArray(payloads.mutationClassId, mcIds.slice(i, i + 50)))) {
              orbits.set(b.id, b.labeled);
            }
          }
          for (const r of page) {
            const labs = r.mcId
              ? (orbits.get(r.mcId) ?? []).filter((e) => e.qmd_id === r.id)
              : [];
            if (labs.length === 0) {
              chunk += csvLine(exportRow(r));
              rowCount += 1;
            } else {
              for (const e of labs) {
                chunk += csvLine(exportRow(r, e.matrix));
                rowCount += 1;
              }
            }
          }
        } else {
          for (const r of page) {
            chunk += csvLine(exportRow(r));
            rowCount += 1;
          }
        }
        await writer.write(encoder.encode(chunk));
        if (page.length < PAGE) break;
      }
      await writer.close();
    } catch (err) {
      await writer.abort(err);
      throw err;
    } finally {
      // Best-effort logging: a tracking failure must never break the download.
      try {
        await db.insert(downloads).values({
          createdAt: new Date().toISOString().replace("T", " ").slice(0, 19),
          fmt, rowCount, filters: loggedFilters, email, name,
          ip, userAgent, referer,
        });
      } catch (e) {
        console.error("download logging failed", e);
      }
    }
  };
  c.executionCtx.waitUntil(pump());

  const iso = new Date().toISOString();   // e.g. 2026-08-04T20:26:33.123Z
  const stamp = iso.slice(0, 10).replaceAll("-", "")
    + "-" + iso.slice(11, 19).replaceAll(":", "");
  const base = scope === "labelings" ? "qmd-labelings" : "qmd-quivers";
  return new Response(readable, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `attachment; filename="${base}-${stamp}.csv"`,
    },
  });
}

exportRoutes.get("/export", handleExport);
// Spelled route from the migration brief; same handler, same output.
exportRoutes.get("/export.csv", handleExport);
