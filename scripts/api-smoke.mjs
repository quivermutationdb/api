#!/usr/bin/env node
/**
 * API smoke tests, run against a wrangler dev instance whose local D1 holds
 * the full exported dataset (see README "Loading data"):
 *
 *     npm run dev            # terminal 1
 *     npm run test:api       # terminal 2 (or: node scripts/api-smoke.mjs)
 *
 * Asserts the frontend contract (field names, envelope, 404 shapes) and
 * cross-checks known mathematical facts (A2/A3/D4 class sizes, Markov
 * non-mutation-acyclicity) so a wrong data load fails loudly, not quietly.
 */

const BASE = process.env.QMD_API ?? "http://127.0.0.1:8787/api";

let failures = 0;
function check(name, cond, extra = "") {
  if (cond) console.log(`  PASS  ${name}`);
  else { failures += 1; console.error(`  FAIL  ${name}${extra ? " — " + extra : ""}`); }
}

async function get(path, expectStatus = 200) {
  const res = await fetch(BASE + path);
  if (res.status !== expectStatus) {
    throw new Error(`${path}: HTTP ${res.status}, expected ${expectStatus}`);
  }
  return res;
}
const json = async (path, s) => (await get(path, s)).json();

// ---- /health, /stats ------------------------------------------------------
{
  const h = await json("/health");
  check("/health ok", h.status === "ok");

  const s = await json("/stats");
  check("/stats totals", s.distinct_quivers === 724 && s.labeled_quivers === 3754
    && s.mutation_classes === 178, JSON.stringify(s));
  check("/stats by_rank", Array.isArray(s.by_rank) && s.by_rank.length === 4
    && s.by_rank[3].distinct_quivers === 695);
}

// ---- /quivers envelope + filters ------------------------------------------
{
  const d = await json("/quivers?limit=5");
  check("/quivers envelope", d.items.length === 5 && d.total === 724
    && d.distinct_total === 724 && d.labeled_total === 3754,
    JSON.stringify({ total: d.total, dt: d.distinct_total, lt: d.labeled_total }));
  const item = d.items[0];
  const keys = ["qmd_id", "num_vertices", "dynkin_type", "representation_type",
    "max_edge", "is_acyclic", "is_connected", "is_bipartite", "is_open",
    "class_size", "exchange_matrix", "mc_id"];
  check("/quivers item keys", keys.every((k) => k in item),
    Object.keys(item).join(","));
  check("/quivers default sort n asc", d.items[0].num_vertices === 1);

  const r4 = await json("/quivers?rank=4&limit=1");
  check("/quivers rank filter", r4.total === 695 && r4.items[0].num_vertices === 4);

  const open = await json("/quivers?rank=3&is_open=true&limit=1");
  const closed = await json("/quivers?rank=3&is_open=false&limit=1");
  check("/quivers is_open partitions rank 3",
    open.total + closed.total === 25, `${open.total}+${closed.total}`);
  check("open class_size is null (∞)", open.items[0].class_size === null);

  const desc = await json("/quivers?sort=class_size&dir=desc&limit=1");
  check("/quivers sort=class_size desc", desc.items[0].class_size === null
    || typeof desc.items[0].class_size === "number");

  const bad = await get("/quivers?rank=abc", 400);
  check("/quivers bad rank -> 400 detail", "detail" in await bad.json());
}

// ---- labelings scope -------------------------------------------------------
{
  const d = await json("/quivers?rank=3&scope=labelings&limit=10&offset=0");
  check("labelings scope total", d.total === 56 && d.labeled_total === 56,
    JSON.stringify({ total: d.total }));
  // Page through all labelings of rank 3 and confirm the count adds up.
  let n = 0;
  for (let off = 0; off < 56; off += 10) {
    const page = await json(`/quivers?rank=3&scope=labelings&limit=10&offset=${off}`);
    n += page.items.length;
  }
  check("labelings pagination covers all", n === 56, String(n));
}

// ---- /search ---------------------------------------------------------------
{
  const mf = await json("/search?is_mutation_finite=true&rank=3");
  const open = await json("/search?is_open=true&rank=3");
  check("/search mutation_finite + open partition rank 3",
    mf.total + open.total === 25, `${mf.total}+${open.total}`);

  const laced = await json("/search?rank=3&is_simply_laced=true");
  check("/search is_simply_laced", laced.items.every((i) => i.max_edge <= 1));

  const a3 = await json("/search?dynkin_type=A3");
  check("/search dynkin A3", a3.total >= 1
    && a3.items.every((i) => i.dynkin_type === "A3" && i.class_size === 14));
}

// ---- quiver + class detail -------------------------------------------------
{
  const a3list = await json("/search?dynkin_type=A3&limit=1");
  const qid = a3list.items[0].qmd_id;
  const mcId = a3list.items[0].mc_id;

  const qd = await json(`/quivers/${qid}`);
  check("quiver detail shape", qd.qmd_id === qid && qd.label === "A3"
    && qd.class_size === 14 && Array.isArray(qd.exchange_matrix)
    && Array.isArray(qd.tags) && "symmetry_group" in qd && "is_planar" in qd);

  const cd = await json(`/classes/${mcId}`);
  check("class detail shape", cd.mc_id === mcId && cd.labeled_size === 14
    && cd.distinct_quiver_count === 4 && cd.labeled_quivers.length === 14
    && cd.distinct_quivers.length === 4 && "provenance" in cd
    && cd.size_of_explored_mutation_class === 14);
  check("class detail canonical first", cd.distinct_quivers[0].is_canonical === true
    && cd.distinct_quivers[0].qmd_id === cd.canonical_qid);
  const labelSum = cd.distinct_quivers.reduce((a, d) => a + d.labeling_count, 0);
  check("class detail labeling counts sum", labelSum === 14, String(labelSum));

  // Markov quiver: rank 3, closed, proved NOT mutation-acyclic.
  const rank3 = await json("/classes?rank=3");
  const markov = rank3.items.find((i) => i.is_mutation_acyclic === false);
  check("/classes browse finds Markov (is_mutation_acyclic=false)", !!markov);

  check("quiver 404", "detail" in await (await get("/quivers/Q.n3.doesnotexist00", 404)).json());
  check("class 404", "detail" in await (await get("/classes/MC.n3.doesnotexist0", 404)).json());
  check("malformed id 404", "detail" in await (await get("/quivers/garbage", 404)).json());
}

// ---- /classes browse -------------------------------------------------------
{
  const d = await json("/classes?rank=3");
  check("/classes rank 3 total", d.total === 11, String(d.total));
  const a2 = await json("/classes?dynkin_type=A2");
  check("/classes A2 class_size", a2.total === 1 && a2.items[0].class_size === 2);
  const d4 = await json("/classes?dynkin_type=D4");
  check("/classes D4 class_size", d4.total === 1 && d4.items[0].class_size === 50);
}

// ---- /random ---------------------------------------------------------------
{
  const rq = await json("/random/quiver");
  check("/random/quiver", /^Q\.n\d+\./.test(rq.qmd_id));
  const rc = await json("/random/class");
  check("/random/class", /^MC\.n\d+\./.test(rc.mc_id));
  const detail = await json(`/quivers/${rq.qmd_id}`);
  check("random quiver resolves", detail.qmd_id === rq.qmd_id);
}

// ---- /export ---------------------------------------------------------------
{
  const res = await get("/export?rank=2");
  const disp = res.headers.get("content-disposition") ?? "";
  check("export headers", res.headers.get("content-type")?.includes("text/csv") === true
    && disp.includes("qmd-quivers-"), disp);
  // res.text() strips a leading BOM per the WHATWG spec — check raw bytes.
  const bytes = new Uint8Array(await res.arrayBuffer());
  check("export BOM", bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf);
  const text = new TextDecoder().decode(bytes);
  const lines = text.split("\r\n").filter((l) => l.length > 0);
  check("export header row", lines[0].endsWith("qmd_id,num_vertices,exchange_matrix,"
    + "representation_type,max_edge,is_acyclic,is_connected,is_bipartite,"
    + "is_abundant,is_planar,symmetry_order,symmetry_name,mc_id,dynkin_type,"
    + "is_open,class_size,labeled_size,distinct_quiver_count,"
    + "merged_orbit_count,is_finite_confirmed,is_infinite_confirmed,"
    + "is_infinite_expected,size_of_explored_frontier,is_mutation_acyclic,"
    + "is_banff,is_louise,is_p_prime"));
  check("export rank 2 rows", lines.length === 1 + 3, String(lines.length));
  check("export booleans TRUE/FALSE", /,(TRUE|FALSE),/.test(lines[1]));
  check("export matrix quoted JSON", lines[1].includes('"[[0,'));

  const lab = await (await get("/export?rank=2&scope=labelings")).text();
  const labLines = lab.split("\r\n").filter((l) => l.length > 0);
  check("export labelings rank 2 rows", labLines.length === 1 + 5, String(labLines.length));

  check("export xlsx rejected", "detail" in await (await get("/export?format=xlsx", 400)).json());
  const alias = await get("/export.csv?rank=2");
  check("/export.csv alias works", (await alias.text()).split("\r\n").length
    === text.split("\r\n").length);
}

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
