/**
 * /mcp — a stateless Model Context Protocol server over Streamable HTTP
 * (no auth; the dataset is public and read-only), so any MCP-capable agent
 * can query QMD directly: https://quivermutationdb.org/mcp
 *
 * Tools wrap the same functions the REST API uses, so results are identical
 * and every list is keyset-paged (pass `cursor` back to continue).
 */

import { McpServer } from "@modelcontextprotocol/server";
import { createMcpHandler } from "agents/mcp/server";
import { asc } from "drizzle-orm";
import { z } from "zod";
import { classDetail, classLabelings, classListParamsFrom, classQuivers, listClasses, slugToId } from "./api/classes";
import { listParamsFrom, listQuivers, quiverDetail, quiverLabelings } from "./api/quivers";
import { classNicknames as nick, mutationClasses as mc, rankStats } from "./db/schema";
import { dbForId, mainDb } from "./db/shard";
import { eq } from "drizzle-orm";
import { lookupMatrix } from "./api/lookup";

const json = (v: unknown) => ({ content: [{ type: "text" as const, text: JSON.stringify(v) }] });
const fail = (msg: string) => ({ content: [{ type: "text" as const, text: msg }], isError: true });

/** Turn a tool's args into the (k) => string|undefined getter the REST parsers expect. */
function getter(args: Record<string, unknown>) {
  return (k: string) => {
    const v = args[k];
    return v === undefined || v === null ? undefined : String(v);
  };
}

const FILTERS = {
  rank: z.number().int().optional().describe("number of vertices"),
  dynkin_type: z.string().optional().describe('finite cluster type, e.g. "A3", "D4", "A1 + A2"'),
  representation_type: z.enum(["finite", "tame", "wild"]).optional(),
  max_edge: z.number().int().optional().describe("largest |b_ij|"),
  is_open: z.boolean().optional().describe("class only partially explored"),
  orbit_min: z.number().int().optional(), orbit_max: z.number().int().optional(),
  is_acyclic: z.boolean().optional(), is_connected: z.boolean().optional(),
  is_simply_laced: z.boolean().optional(),
  is_mutation_finite: z.boolean().optional().describe("true = proved finite, false = proved infinite"),
  nickname: z.string().optional().describe("curated nickname slug, e.g. markov"),
};
const PAGING = {
  limit: z.number().int().min(1).max(1000).optional(),
  cursor: z.string().optional().describe("next_cursor from a previous call"),
};

export function createQmdServer(env: Env) {
  const server = new McpServer({ name: "Quiver Mutation Database", version: "2.0.0" });

  server.registerTool("get_stats", {
    description: "Dataset totals and, per rank, how the data was generated (weight bound, node cap, pipeline version, date).",
    inputSchema: {},
  }, async () => {
    const rows = await mainDb(env).select().from(rankStats).orderBy(asc(rankStats.n));
    return json({
      distinct_quivers: rows.reduce((a, r) => a + r.quiverCount, 0),
      labeled_quivers: rows.reduce((a, r) => a + r.labeledQuiverCount, 0),
      mutation_classes: rows.reduce((a, r) => a + r.classCount, 0),
      by_rank: rows.map((r) => ({ n: r.n, distinct_quivers: r.quiverCount, labeled_quivers: r.labeledQuiverCount,
        mutation_classes: r.classCount, bound: r.bound, node_cap: r.nodeCap, generated_at: r.generatedAt,
        pipeline_version: r.pipelineVersion, generator: r.generator, census_size: r.censusSize })),
    });
  });

  server.registerTool("search_quivers", {
    description: "Filter, sort and page quivers. Returns items with exchange_matrix and class summary plus next_cursor. "
      + "scope=labelings returns one row per labeled exchange matrix (default sort only).",
    inputSchema: {
      ...FILTERS, ...PAGING,
      scope: z.enum(["distinct", "labelings"]).optional(),
      sort: z.enum(["qmd_id", "num_vertices", "class_size", "max_edge", "dynkin_type", "class_type"]).optional(),
      dir: z.enum(["asc", "desc"]).optional(),
    },
  }, async (args) => {
    try {
      const p = listParamsFrom(getter(args), 50);
      return json(await listQuivers(env, p));
    } catch (e) { return fail(String((e as Error).message)); }
  });

  server.registerTool("get_quiver", {
    description: "One quiver by id (Q.n{rank}.{hash}): canonical exchange matrix, invariants, class summary.",
    inputSchema: { id: z.string() },
  }, async ({ id }) => {
    const d = await quiverDetail(env, id);
    return d ? json(d) : fail(`No quiver ${id}`);
  });

  server.registerTool("get_quiver_labelings", {
    description: "Every labeled exchange matrix of a quiver within its mutation class (paged).",
    inputSchema: { id: z.string(), ...PAGING },
  }, async ({ id, limit, cursor }) => {
    const db = dbForId(env, id);
    const row = db ? (await db.select({ mcId: mc.id }).from(mc).where(eq(mc.id, id)))[0] : undefined;
    const qrow = db ? (await db.select({ mcId: (await import("./db/schema")).quivers.mutationClassId }).from((await import("./db/schema")).quivers).where(eq((await import("./db/schema")).quivers.id, id)))[0] : undefined;
    if (!db || !qrow) return fail(`No quiver ${id}`);
    void row;
    return json({ qmd_id: id, ...(await quiverLabelings(env, id, qrow.mcId, cursor, limit ?? 100)) });
  });

  server.registerTool("list_classes", {
    description: "Filter, sort and page mutation classes (with nickname, exploration state, three-state properties).",
    inputSchema: {
      rank: z.number().int().optional(), dynkin_type: z.string().optional(), is_open: z.boolean().optional(),
      is_mutation_finite: z.boolean().optional(), is_mutation_acyclic: z.boolean().optional(),
      orbit_min: z.number().int().optional(), orbit_max: z.number().int().optional(), nickname: z.string().optional(),
      sort: z.enum(["mc_id", "num_vertices", "class_size", "distinct_quiver_count", "dynkin_type", "class_type"]).optional(),
      dir: z.enum(["asc", "desc"]).optional(), ...PAGING,
    },
  }, async (args) => {
    try {
      const p = classListParamsFrom(getter(args));
      return json(await listClasses(env, p));
    } catch (e) { return fail(String((e as Error).message)); }
  });

  server.registerTool("get_class", {
    description: "One mutation class by id (MC.n{rank}.{hash}) or curated nickname slug (e.g. markov): invariants with "
      + "three-state values, provenance/witnesses, canonical matrix, first page of distinct member quivers.",
    inputSchema: { id_or_slug: z.string() },
  }, async ({ id_or_slug }) => {
    let id = id_or_slug;
    if (!/^MC\.n\d+\./.test(id)) {
      const resolved = await slugToId(env, id);
      if (!resolved) return fail(`No class with nickname ${id}`);
      id = resolved;
    }
    const d = await classDetail(env, id);
    return d ? json(d) : fail(`No class ${id}`);
  });

  server.registerTool("list_class_members", {
    description: "Members of a mutation class, paged: kind=distinct (one per unlabeled quiver, canonical first) or "
      + "kind=labelings (every labeled exchange matrix in orbit order; qmd_id restricts to one quiver).",
    inputSchema: { id: z.string(), kind: z.enum(["distinct", "labelings"]).optional(), qmd_id: z.string().optional(), ...PAGING },
  }, async ({ id, kind, qmd_id, limit, cursor }) => {
    const db = dbForId(env, id);
    if (!db) return fail(`No class ${id}`);
    const row = (await db.select({ canon: mc.canonicalQuiverId }).from(mc).where(eq(mc.id, id)))[0];
    if (!row) return fail(`No class ${id}`);
    const lim = limit ?? 100;
    return json({ mc_id: id, ...(kind === "labelings"
      ? await classLabelings(env, id, qmd_id, cursor, lim)
      : await classQuivers(env, id, row.canon, cursor, lim)) });
  });

  server.registerTool("lookup_quiver", {
    description: "Find the quiver a skew-symmetric exchange matrix represents: returns its canonical (lex-min) "
      + "matrix and Q.* id, and the database row if the census contains it (found=false otherwise). "
      + "b_ij > 0 means b_ij arrows i -> j.",
    inputSchema: { matrix: z.array(z.array(z.number().int())).describe("square skew-symmetric integer matrix") },
  }, async ({ matrix }) => {
    try { return json(await lookupMatrix(env, matrix)); } catch (e) { return fail(String((e as Error).message)); }
  });

  server.registerTool("list_nicknames", {
    description: "Curated class nicknames (slug -> class id).",
    inputSchema: {},
  }, async () => {
    const rows = await mainDb(env).select().from(nick).orderBy(asc(nick.slug));
    return json({ items: rows.map((r) => ({ mc_id: r.mcId, nickname: r.nickname, slug: r.slug, note: r.note })) });
  });

  return server;
}

export function mcpHandler(env: Env) {
  return createMcpHandler(() => createQmdServer(env), { route: "/mcp" });
}
