#!/usr/bin/env node
/**
 * API smoke tests, run against a wrangler dev instance whose local D1 holds
 * the v3 export of the connected n<=4 census (see README "Loading data"):
 *
 *     npm run dev            # terminal 1
 *     npm run test:api       # terminal 2
 *
 * Asserts the frontend contract (field names, envelope, 404 shapes) and
 * cross-checks mathematical facts (A3/D4 class sizes, Markov, E-free rank 4,
 * Derksen–Owen labels) so a wrong data load fails loudly. Totals are taken
 * from /stats rather than hard-coded, except the exact connected census sizes.
 */

const BASE = process.env.QMD_API ?? "http://127.0.0.1:8787/api";

let failures = 0;
function check(name, cond, extra = "") {
  if (cond) console.log(`  PASS  ${name}`);
  else { failures += 1; console.error(`  FAIL  ${name}${extra ? " — " + extra : ""}`); }
}
async function get(path, expectStatus = 200) {
  const res = await fetch(BASE + path);
  if (res.status !== expectStatus) throw new Error(`${path}: HTTP ${res.status}, expected ${expectStatus}`);
  return res;
}
const json = async (path, s) => (await get(path, s)).json();
const monotone = (xs, cmp) => xs.every((x, i) => i === 0 || cmp(xs[i - 1], x) <= 0);
async function walk(path, pick) {
  const out = []; let cursor = "";
  for (;;) {
    const d = await json(`${path}${path.includes("?") ? "&" : "?"}limit=100${cursor ? "&cursor=" + encodeURIComponent(cursor) : ""}`);
    out.push(...d.items.map(pick));
    if (!d.next_cursor) return out;
    cursor = d.next_cursor;
  }
}

// ---- /health, /stats ------------------------------------------------------
const st = await json("/stats");
{
  check("/health ok", (await json("/health")).status === "ok");
  const byRank = Object.fromEntries(st.by_rank.map((r) => [r.n, r]));
  check("/stats connected census sizes (1,2,22,667)", byRank[1].distinct_quivers === 1 && byRank[2].distinct_quivers === 2
    && byRank[3].distinct_quivers === 22 && byRank[4].distinct_quivers === 667, JSON.stringify(st.by_rank.map((r) => r.distinct_quivers)));
  check("/stats census_size == distinct (exact census)", st.by_rank.every((r) => r.census_size === r.distinct_quivers && r.generator === "orderly"));
  check("/stats provenance", st.by_rank.every((r) => r.pipeline_version === "2.0.0" && r.bound === 2 && r.shard_counts));
  check("/stats totals sum", st.distinct_quivers === 692 && st.mutation_classes === st.by_rank.reduce((a, r) => a + r.mutation_classes, 0));
}

// ---- /quivers envelope + filters ------------------------------------------
{
  const d = await json("/quivers?limit=5");
  check("/quivers envelope", d.items.length === 5 && d.total === st.distinct_quivers && d.distinct_total === st.distinct_quivers);
  const item = d.items[0];
  const keys = ["qmd_id", "num_vertices", "dynkin_type", "representation_type", "max_edge", "is_acyclic",
    "is_connected", "is_bipartite", "is_open", "exploration", "explored", "mutation_finite", "class_size",
    "explored_size", "exchange_matrix", "mc_id", "nickname"];
  check("/quivers item keys", keys.every((k) => k in item), Object.keys(item).join(","));
  check("/quivers matrices decoded", Array.isArray(item.exchange_matrix) && item.exchange_matrix.length === item.num_vertices);
  check("/quivers default sort n asc", d.items[0].num_vertices === 1);
  check("all quivers connected", (await json("/quivers?is_connected=false&limit=1")).total === 0);
  const r4 = await json("/quivers?rank=4&limit=1");
  check("/quivers rank filter", r4.total === 667 && r4.items[0].num_vertices === 4);
  const open = await json("/quivers?rank=3&is_open=true&limit=100");
  const closed = await json("/quivers?rank=3&is_open=false&limit=100");
  check("rank-3 open/closed partition", open.total + closed.total === 22 && open.items.every((i) => i.is_open) && closed.items.every((i) => !i.is_open));
  const fin = await json("/quivers?is_mutation_finite=true&limit=1000");
  const inf = await json("/quivers?is_mutation_finite=false&limit=1000");
  check("mutation_finite filter uses the per-quiver label", fin.items.every((i) => i.mutation_finite === true) && inf.items.every((i) => i.mutation_finite === false));
  check("every rank-3/4 quiver has a finiteness label (bound 2 explores to the wall)", fin.total + inf.total === st.distinct_quivers, `${fin.total}+${inf.total}`);
  check("simply-laced filter", (await json("/quivers?is_simply_laced=true&limit=1000")).items.every((i) => i.max_edge <= 1));
  check("explored filter", (await json("/quivers?explored=true&limit=1")).total === st.distinct_quivers && (await json("/quivers?explored=false&limit=1")).total === 0);
  const a3 = await json("/search?dynkin_type=A3&limit=5");
  check("A3 class size 14, D4 50", a3.items[0].class_size === 14 && (await json("/search?dynkin_type=D4&limit=1")).items[0].class_size === 50);
  check("bad int -> 400", "detail" in await json("/quivers?rank=abc", 400));
  check("unknown sort -> 400", "detail" in await json("/quivers?sort=bogus", 400));
  check("bad dir -> 400", "detail" in await json("/quivers?dir=sideways", 400));
  check("bad cursor -> 400", "detail" in await json("/quivers?cursor=kZZZ", 400));
}

// ---- sorting + cursors ------------------------------------------------------
{
  const ids = await walk("/quivers?rank=4", (i) => i.qmd_id);
  check("cursor walk covers rank 4 exactly, no dups", ids.length === 667 && new Set(ids).size === 667, String(ids.length));
  const me = await json("/quivers?rank=4&sort=max_edge&dir=desc&limit=100");
  check("sort max_edge desc monotone", monotone(me.items.map((i) => i.max_edge), (a, b) => b - a));
  const dt = await walk("/quivers?rank=4&sort=dynkin_type&dir=desc&is_open=false", (i) => i.dynkin_type ?? "");
  check("cursor walk under dynkin desc with NULLs covers all finite rank-4 quivers",
    dt.length === (await json("/quivers?rank=4&is_open=false&limit=1")).total && monotone(dt, (a, b) => a < b ? 1 : a > b ? -1 : 0));
  const cs = await json("/quivers?rank=4&sort=class_size&dir=desc&is_open=false&limit=100");
  check("sort class_size desc monotone (nulls last)", monotone(cs.items.map((i) => i.class_size), (a, b) => (b ?? -1) - (a ?? -1)));
  check("offset paging still works", (await json("/quivers?rank=4&offset=660&limit=100")).items.length === 7);
  check("labelings scope rejects custom sort", "detail" in await json("/quivers?scope=labelings&sort=max_edge", 400));
  const labs = await walk("/quivers?rank=3&scope=labelings", (i) => i.labeling_ord);
  check("labelings scope walk = rank-3 stored labelings (28)", labs.length === st.by_rank.find((r) => r.n === 3).labeled_quivers && labs.length === 28, String(labs.length));
}

// ---- classes: list, detail, members --------------------------------------
{
  const all = await json("/classes?limit=1");
  check("/classes total from rank_stats", all.total === st.mutation_classes);
  const a3 = await json("/search?dynkin_type=A3&limit=1");
  const cls = await json(`/classes/${a3.items[0].mc_id}`);
  check("class detail: A3 complete, 4 distinct, 14 labelings inline, canonical first",
    cls.exploration === "complete" && cls.distinct_quiver_count === 4 && cls.distinct_quivers[0].is_canonical
    && cls.labeled_quivers.length === 14 && cls.labelings_stored && !cls.labeled_quivers_truncated);
  check("class detail matrices decoded", Array.isArray(cls.canonical_matrix) && cls.canonical_matrix.length === 3);
  const mem = await json(`/classes/${a3.items[0].mc_id}/quivers?limit=2`);
  const mem2 = await json(`/classes/${a3.items[0].mc_id}/quivers?limit=2&cursor=${encodeURIComponent(mem.next_cursor)}`);
  const ids = [...mem.items, ...mem2.items].map((i) => i.qmd_id);
  check("class members paged, canonical pinned, no dup", new Set(ids).size === 4 && mem.items[0].is_canonical && !mem2.next_cursor, ids.join(","));
  const labs = await json(`/classes/${a3.items[0].mc_id}/labelings?limit=5`);
  const labs2 = await json(`/classes/${a3.items[0].mc_id}/labelings?limit=5&cursor=${encodeURIComponent(labs.next_cursor)}`);
  check("class labelings paged by ord", labs.items.length === 5 && labs2.items[0].ord === 5);
  const openCls = await json("/classes?rank=4&is_open=true&limit=1");
  const oc = await json(`/classes/${openCls.items[0].mc_id}`);
  check("open class: no labelings stored, class_size null, infinite confirmed",
    oc.labelings_stored === false && oc.labeled_size === null && oc.is_infinite_confirmed === true && oc.exploration === "bound"
    && (await json(`/classes/${oc.mc_id}/labelings`)).items.length === 0);
  const finC = await json("/classes?is_mutation_finite=true&limit=1000");
  const infC = await json("/classes?is_mutation_finite=false&limit=1000");
  check("class finiteness filters partition", finC.total + infC.total === st.mutation_classes && finC.items.every((c) => c.is_finite_confirmed));
  check("class sort dynkin desc + unknown sort 400", "detail" in await json("/classes?sort=bogus", 400)
    && monotone((await json("/classes?rank=4&sort=dynkin_type&dir=desc&is_open=false&limit=100")).items.map((c) => c.dynkin_type ?? ""), (a, b) => a < b ? 1 : a > b ? -1 : 0));
  check("404 shapes", "detail" in await json("/quivers/Q.n3.0000000000000000", 404) && "detail" in await json("/classes/nope", 404));
}

// ---- Markov + nicknames + Derksen–Owen -----------------------------------
{
  const mk = await json("/classes/by-slug/markov");
  check("markov by slug: closed 2-element labeled orbit, not mutation-acyclic",
    mk.mc_id === "MC.n3.7405511b230b7552" && mk.nickname === "Markov" && mk.labeled_size === 2 && mk.is_mutation_acyclic === false);
  check("/nicknames", (await json("/nicknames")).items.some((i) => i.slug === "markov"));
  check("quiver rows carry nickname", (await json("/quivers?nickname=markov")).items.every((i) => i.nickname === "Markov"));
  const ql = await json(`/quivers/${mk.canonical_qid}/labelings`);
  check("quiver labelings endpoint", ql.items.length === 2 && ql.items.every((l) => l.mc_id === mk.mc_id));
  const rq = await json("/random/quiver"); const rc = await json("/random/class");
  check("random endpoints", /^Q\.n\d+\./.test(rq.qmd_id) && /^MC\.n\d+\./.test(rc.mc_id) && (await json(`/quivers/${rq.qmd_id}`)).qmd_id === rq.qmd_id);
}

// ---- matrix lookup ---------------------------------------------------------
{
  const a3 = await json("/lookup?matrix=" + encodeURIComponent("[[0,-1,0],[1,0,1],[0,-1,0]]"));   // a relabeling/reorientation of A3
  check("lookup canonicalises to a census quiver", a3.found === true && a3.qmd_id.startsWith("Q.n3.") && a3.quiver.dynkin_type === "A3");
  const mk = await json("/lookup?matrix=" + encodeURIComponent("[[0,-2,2],[2,0,-2],[-2,2,0]]"));
  check("lookup finds Markov from any labeling", mk.found && mk.quiver.nickname === "Markov");
  const big = await json("/lookup?matrix=" + encodeURIComponent("[[0,3,0],[-3,0,1],[0,-1,0]]"));
  check("lookup outside the census: found=false with canonical id", big.found === false && /^Q\.n3\.[0-9a-f]{16}$/.test(big.qmd_id) && big.max_edge === 3);
  check("lookup rejects non-skew", "detail" in await json("/lookup?matrix=" + encodeURIComponent("[[0,1],[1,0]]"), 400));
  const post = await (await fetch(BASE + "/lookup", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ matrix: [[0, 1], [-1, 0]] }) })).json();
  check("lookup POST", post.found === true && post.num_vertices === 2);
}

// ---- export: csv + ndjson -----------------------------------------------
{
  const res = await get("/export?rank=2");
  const bytes = new Uint8Array(await res.arrayBuffer());
  check("export BOM + headers", bytes[0] === 0xef && res.headers.get("content-type")?.includes("text/csv") === true);
  const lines = new TextDecoder().decode(bytes).split("\r\n").filter((l) => l.length > 0);
  check("export header row", lines[0].endsWith("is_banff,is_louise,is_p_prime,exploration,nickname,mutation_finite,explored"));
  check("export rank 2 rows (2 connected quivers)", lines.length === 3, String(lines.length));
  check("export matrix quoted JSON + TRUE/FALSE", lines[1].includes('"[[0,') && /,(TRUE|FALSE),/.test(lines[1]));
  const lab = (await (await get("/export?rank=2&scope=labelings")).text()).split("\r\n").filter((l) => l.length > 0);
  check("export labelings rank 2 rows (4)", lab.length === 5, String(lab.length));
  check("export xlsx rejected", "detail" in await (await get("/export?format=xlsx", 400)).json());
  const r1 = await get("/export.ndjson?rank=3&limit=10");
  const l1 = (await r1.text()).trim().split("\n");
  const next = r1.headers.get("x-next-cursor");
  check("ndjson page 1", l1.length === 10 && JSON.parse(l1[0]).qmd_id.startsWith("Q.n3.") && next);
  const r2 = await get(`/export.ndjson?rank=3&limit=100&cursor=${encodeURIComponent(next)}`);
  check("ndjson resume completes rank 3", (await r2.text()).trim().split("\n").length === 12 && r2.headers.get("x-next-cursor") === "");
  const full = (await (await get("/export.ndjson")).text()).trim().split("\n");
  check("ndjson full stream = all quivers", full.length === st.distinct_quivers, String(full.length));
}

// ---- openapi, cors, cache ---------------------------------------------------
{
  const spec = await json("/openapi.json");
  const paths = Object.keys(spec.paths);
  check("openapi paths", ["/quivers", "/classes/{id}/quivers", "/export.ndjson", "/nicknames", "/lookup", "/stats"].every((p) => paths.includes(p)));
  check("CORS open", (await get("/health")).headers.get("access-control-allow-origin") === "*");
  check("lists are cacheable", (await get("/quivers?limit=1")).headers.get("cache-control") === "public, max-age=300");
}

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
