/**
 * GET /openapi.json — OpenAPI 3.1 description of the public API, for
 * tool-calling agents and generated clients. Hand-maintained: keep it in step
 * with the routes (scripts/api-smoke.mjs checks that every path here exists).
 */

import { Hono } from "hono";

const FILTER_PARAMS = [
  p("rank", "integer", "Number of vertices (rank)."),
  p("dynkin_type", "string", "Finite cluster type label of the class, e.g. A3, D4, \"A1 + A2\"."),
  p("representation_type", "string", "finite | tame | wild (acyclic quivers only)."),
  p("max_edge", "integer", "Largest |b_ij| of the quiver."),
  p("is_open", "boolean", "Class only partially explored (exploration != complete)."),
  p("orbit_min", "integer", "Minimum explored class size (labeled matrices)."),
  p("orbit_max", "integer", "Maximum explored class size."),
  p("is_acyclic", "boolean", "Quiver has no oriented cycle."),
  p("is_connected", "boolean", "Underlying graph is connected."),
  p("is_simply_laced", "boolean", "All |b_ij| <= 1."),
  p("is_mutation_finite", "boolean", "true: proved mutation-finite; false: proved mutation-infinite. Undetermined classes match neither."),
  p("nickname", "string", "Curated class nickname slug, e.g. markov."),
];
const PAGE_PARAMS = [
  p("limit", "integer", "Page size (max 1000)."),
  p("offset", "integer", "Row offset for page-number UIs. Prefer cursor for bulk reads."),
  p("cursor", "string", "Opaque keyset cursor from a previous response's next_cursor. Overrides offset."),
];
const SORT_PARAMS = [
  p("sort", "string", "qmd_id | num_vertices | class_size | max_edge | dynkin_type | class_type"),
  p("dir", "string", "asc | desc"),
];

function p(name: string, type: string, description: string) {
  return { name, in: "query", required: false, schema: { type }, description };
}
function pathParam(name: string, description: string) {
  return { name, in: "path", required: true, schema: { type: "string" }, description };
}
function ok(description: string, schemaRef: string) {
  return { 200: { description, content: { "application/json": { schema: { $ref: `#/components/schemas/${schemaRef}` } } } } };
}
const notFound = { 404: { description: "Not found", content: { "application/json": { schema: { $ref: "#/components/schemas/Error" } } } } };
const badRequest = { 400: { description: "Bad query parameter (unparseable value, unknown sort, bad cursor)", content: { "application/json": { schema: { $ref: "#/components/schemas/Error" } } } } };

export const OPENAPI = {
  openapi: "3.1.0",
  info: {
    title: "Quiver Mutation Database API",
    version: "2.0.0",
    summary: "Quivers, exchange matrices, and mutation classes with rigorously defined invariants.",
    description:
      "Read-only, no authentication, CORS open. Identifiers: a quiver is `Q.n{rank}.{sha256[:16]}` "
      + "(hash of its lex-min canonical exchange matrix), a mutation class is `MC.n{rank}.{...}` "
      + "(hash of the lex-min matrix over the whole explored orbit). Every list response carries "
      + "`next_cursor`; pass it back as `?cursor=` to page. For bulk pulls use /export.ndjson. "
      + "Semidecidable properties are three-state: true / false / null (unknown), never guessed. "
      + "Definitions: https://quivermutationdb.org/wiki. Licence CC-BY-4.0 — please cite.",
    license: { name: "CC-BY-4.0", url: "https://creativecommons.org/licenses/by/4.0/" },
    contact: { name: "Blake Jackson", email: "jackson@icarm.io" },
  },
  servers: [{ url: "https://quivermutationdb.org/api" }],
  paths: {
    "/stats": { get: { summary: "Dataset totals and per-rank generation provenance", operationId: "getStats", responses: ok("Totals", "Stats") } },
    "/quivers": { get: { summary: "List quivers (browse)", operationId: "listQuivers",
      parameters: [...FILTER_PARAMS, p("scope", "string", "distinct (default) | labelings — one row per labeled matrix; labelings scope supports only the default sort."), ...SORT_PARAMS, ...PAGE_PARAMS],
      responses: { ...ok("Page of quivers", "QuiverList"), ...badRequest } } },
    "/search": { get: { summary: "Same as /quivers with a larger default page (100)", operationId: "searchQuivers",
      parameters: [...FILTER_PARAMS, ...SORT_PARAMS, ...PAGE_PARAMS], responses: { ...ok("Page of quivers", "QuiverList"), ...badRequest } } },
    "/quivers/{id}": { get: { summary: "One quiver with its invariants", operationId: "getQuiver",
      parameters: [pathParam("id", "Q.n{rank}.{hash}")], responses: { ...ok("Quiver", "QuiverDetail"), ...notFound } } },
    "/quivers/{id}/labelings": { get: { summary: "Every labeled exchange matrix of this quiver in its class (paged)", operationId: "getQuiverLabelings",
      parameters: [pathParam("id", "Q.n{rank}.{hash}"), p("limit", "integer", "max 1000"), p("cursor", "string", "keyset cursor")],
      responses: { ...ok("Labelings", "LabelingList"), ...notFound } } },
    "/classes": { get: { summary: "List mutation classes (browse)", operationId: "listClasses",
      parameters: [p("rank", "integer", ""), p("dynkin_type", "string", ""), p("is_open", "boolean", ""), p("is_mutation_finite", "boolean", ""), p("is_mutation_acyclic", "boolean", ""), p("orbit_min", "integer", ""), p("orbit_max", "integer", ""), p("nickname", "string", "slug"),
        p("sort", "string", "mc_id | num_vertices | class_size | distinct_quiver_count | dynkin_type | class_type"), p("dir", "string", "asc | desc"), ...PAGE_PARAMS],
      responses: { ...ok("Page of classes", "ClassList"), ...badRequest } } },
    "/classes/{id}": { get: { summary: "One mutation class: invariants, provenance, first page of members", operationId: "getClass",
      parameters: [pathParam("id", "MC.n{rank}.{hash}")], responses: { ...ok("Class", "ClassDetail"), ...notFound } } },
    "/classes/by-slug/{slug}": { get: { summary: "Class detail by curated nickname slug (e.g. markov)", operationId: "getClassBySlug",
      parameters: [pathParam("slug", "nickname slug")], responses: { ...ok("Class", "ClassDetail"), ...notFound } } },
    "/classes/{id}/quivers": { get: { summary: "Distinct quivers of a class, canonical first then most-labeled (paged)", operationId: "getClassQuivers",
      parameters: [pathParam("id", "MC.n{rank}.{hash}"), p("limit", "integer", "max 1000"), p("cursor", "string", "keyset cursor")],
      responses: { ...ok("Members", "ClassQuiverList"), ...notFound } } },
    "/classes/{id}/labelings": { get: { summary: "Every labeled exchange matrix in the explored orbit (paged)", operationId: "getClassLabelings",
      parameters: [pathParam("id", "MC.n{rank}.{hash}"), p("qmd_id", "string", "restrict to one quiver's labelings"), p("limit", "integer", "max 1000"), p("cursor", "string", "keyset cursor")],
      responses: { ...ok("Labelings", "LabelingList"), ...notFound } } },
    "/random/quiver": { get: { summary: "A uniformly random quiver id", operationId: "randomQuiver", responses: ok("Pick", "RandomQuiver") } },
    "/random/class": { get: { summary: "A uniformly random mutation class id", operationId: "randomClass", responses: ok("Pick", "RandomClass") } },
    "/nicknames": { get: { summary: "Curated class nicknames", operationId: "listNicknames", responses: ok("Nicknames", "NicknameList") } },
    "/export.ndjson": { get: { summary: "Bulk pull of a filtered cut as NDJSON (resumable via X-Next-Cursor)", operationId: "exportNdjson",
      description: "With `limit` (<= 5000): returns up to that many rows and an `X-Next-Cursor` header (empty when done); pass it back as `cursor`. Without `limit`: streams the whole cut. Rows have the CSV export's columns (see EXPORT_COLUMNS in /export).",
      parameters: [...FILTER_PARAMS, p("scope", "string", "distinct | labelings"), p("limit", "integer", "<= 5000; omit to stream everything"), p("cursor", "string", "resume cursor")],
      responses: { 200: { description: "application/x-ndjson", headers: { "X-Next-Cursor": { schema: { type: "string" }, description: "Resume cursor; empty when the cut is exhausted" }, "X-Row-Count": { schema: { type: "integer" } } } }, ...badRequest } } },
    "/export": { get: { summary: "CSV export of a filtered cut (streamed)", operationId: "exportCsv",
      parameters: [...FILTER_PARAMS, p("scope", "string", "distinct | labelings"), p("email", "string", "optional, self-reported for usage tracking"), p("name", "string", "optional")],
      responses: { 200: { description: "text/csv; UTF-8 BOM; CRLF; TRUE/FALSE booleans; empty cell = null" }, ...badRequest } } },
  },
  components: {
    schemas: {
      Error: { type: "object", properties: { detail: { type: "string" } }, required: ["detail"] },
      Matrix: { type: "array", items: { type: "array", items: { type: "integer" } }, description: "Row-major skew-symmetric exchange matrix; b_ij > 0 means b_ij arrows i -> j." },
      TriState: { type: ["boolean", "null"], description: "true / false / null = unknown (search truncated)" },
      Exploration: { type: "string", enum: ["complete", "bound", "truncated"], description: "complete: finite, exact size; bound: a mutation crossed the weight bound (rank >= 3: proves mutation-infinite); truncated: node cap hit, finiteness unknown" },
      Quiver: { type: "object", properties: {
        qmd_id: { type: "string" }, num_vertices: { type: "integer" }, exchange_matrix: { $ref: "#/components/schemas/Matrix" },
        dynkin_type: { type: ["string", "null"] }, representation_type: { type: ["string", "null"] }, max_edge: { type: "integer" },
        is_acyclic: { type: "boolean" }, is_connected: { type: "boolean" }, is_bipartite: { $ref: "#/components/schemas/TriState" },
        is_open: { type: "boolean" }, exploration: { $ref: "#/components/schemas/Exploration" },
        class_size: { type: ["integer", "null"], description: "labeled orbit size when exploration = complete, else null" },
        explored_size: { type: ["integer", "null"] }, mc_id: { type: ["string", "null"] },
        nickname: { type: ["string", "null"] }, nickname_slug: { type: ["string", "null"] },
        labeling_ord: { type: "integer", description: "only in scope=labelings" } } },
      QuiverDetail: { allOf: [{ $ref: "#/components/schemas/Quiver" }, { type: "object", properties: {
        label: { type: ["string", "null"] }, is_abundant: { $ref: "#/components/schemas/TriState" }, is_planar: { $ref: "#/components/schemas/TriState" },
        symmetry_group: { type: ["object", "null"] }, labeling_count: { type: ["integer", "null"] } } }] },
      QuiverList: { type: "object", properties: { items: { type: "array", items: { $ref: "#/components/schemas/Quiver" } }, total: { type: "integer" }, distinct_total: { type: "integer" }, labeled_total: { type: "integer" }, next_cursor: { type: ["string", "null"] } } },
      Class: { type: "object", properties: {
        mc_id: { type: "string" }, label: { type: ["string", "null"] }, nickname: { type: ["string", "null"] }, nickname_slug: { type: ["string", "null"] },
        num_vertices: { type: "integer" }, dynkin_type: { type: ["string", "null"] }, is_open: { type: "boolean" }, exploration: { $ref: "#/components/schemas/Exploration" },
        class_size: { type: ["integer", "null"] }, labeled_size: { type: "integer" }, distinct_quiver_count: { type: "integer" }, merged_orbit_count: { type: "integer" },
        canonical_qid: { type: ["string", "null"] }, is_finite_confirmed: { $ref: "#/components/schemas/TriState" }, is_infinite_confirmed: { $ref: "#/components/schemas/TriState" },
        is_infinite_expected: { $ref: "#/components/schemas/TriState" }, is_mutation_acyclic: { $ref: "#/components/schemas/TriState" },
        is_banff: { $ref: "#/components/schemas/TriState" }, is_louise: { $ref: "#/components/schemas/TriState" }, is_p_prime: { $ref: "#/components/schemas/TriState" } } },
      ClassDetail: { allOf: [{ $ref: "#/components/schemas/Class" }, { type: "object", properties: {
        nickname_note: { type: ["string", "null"] }, canonical_matrix: { $ref: "#/components/schemas/Matrix" },
        distinct_quivers: { type: "array", items: { $ref: "#/components/schemas/ClassQuiver" }, description: "first page (<= 100), canonical first" },
        distinct_quivers_next_cursor: { type: ["string", "null"] },
        labeled_quivers: { type: "array", items: { type: "object", properties: { qmd_id: { type: "string" }, matrix: { $ref: "#/components/schemas/Matrix" } } }, description: "inline only when labeled_size <= 200" },
        labeled_quivers_truncated: { type: "boolean" }, size_of_explored_mutation_class: { type: "integer" }, size_of_explored_frontier: { type: ["integer", "null"] },
        provenance: { type: ["object", "null"], description: "per-property search state and witness" } } }] },
      ClassList: { type: "object", properties: { items: { type: "array", items: { $ref: "#/components/schemas/Class" } }, total: { type: "integer" }, next_cursor: { type: ["string", "null"] } } },
      ClassQuiver: { type: "object", properties: { qmd_id: { type: "string" }, matrix: { $ref: "#/components/schemas/Matrix" }, labeling_count: { type: "integer" }, is_canonical: { type: "boolean" } } },
      ClassQuiverList: { type: "object", properties: { mc_id: { type: "string" }, items: { type: "array", items: { $ref: "#/components/schemas/ClassQuiver" } }, next_cursor: { type: ["string", "null"] } } },
      LabelingList: { type: "object", properties: { items: { type: "array", items: { type: "object", properties: { ord: { type: "integer" }, qmd_id: { type: "string" }, mc_id: { type: "string" }, matrix: { $ref: "#/components/schemas/Matrix" } } } }, next_cursor: { type: ["string", "null"] } } },
      Stats: { type: "object", properties: { distinct_quivers: { type: "integer" }, labeled_quivers: { type: "integer" }, mutation_classes: { type: "integer" },
        by_rank: { type: "array", items: { type: "object", properties: { n: { type: "integer" }, distinct_quivers: { type: "integer" }, labeled_quivers: { type: "integer" }, mutation_classes: { type: "integer" }, bound: { type: ["integer", "null"] }, node_cap: { type: ["integer", "null"] }, generated_at: { type: ["string", "null"] }, pipeline_version: { type: ["string", "null"] }, generator: { type: ["string", "null"], description: "orderly (exact census) | brute | sample" }, census_size: { type: ["integer", "null"], description: "exact number of unlabeled quivers in the cell (n, bound); compare with distinct_quivers to see coverage" } } } } } },
      RandomQuiver: { type: "object", properties: { qmd_id: { type: "string" }, num_vertices: { type: "integer" } } },
      RandomClass: { type: "object", properties: { mc_id: { type: "string" }, num_vertices: { type: "integer" } } },
      NicknameList: { type: "object", properties: { items: { type: "array", items: { type: "object", properties: { mc_id: { type: "string" }, nickname: { type: "string" }, slug: { type: "string" }, note: { type: ["string", "null"] }, num_vertices: { type: ["integer", "null"] }, dynkin_type: { type: ["string", "null"] } } } }, total: { type: "integer" } } },
    },
  },
} as const;

export const openapiRoutes = new Hono<{ Bindings: Env }>();
openapiRoutes.get("/openapi.json", (c) => {
  c.header("Cache-Control", "public, max-age=3600");
  return c.json(OPENAPI);
});
