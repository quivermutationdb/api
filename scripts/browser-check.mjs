import { chromium } from "@playwright/test";
import { writeFileSync } from "node:fs";

const BASE = "http://127.0.0.1:8787";
let failures = 0;
const check = (name, cond, extra = "") => {
  if (cond) console.log(`  PASS  ${name}`);
  else { failures++; console.error(`  FAIL  ${name}${extra ? " — " + extra : ""}`); }
};

// A known A3 quiver/class for the detail pages.
const a3 = await (await fetch(`${BASE}/api/search?dynkin_type=A3&limit=1`)).json();
const qid = a3.items[0].qmd_id;
const mcId = a3.items[0].mc_id;

// PW_CHROMIUM overrides the browser binary (CI sandboxes); default is the
// Playwright-managed Chromium (`npx playwright install chromium`).
const browser = await chromium.launch(
  process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {});
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
// Ignored: external analytics (blocked in sandboxes) and favicon.ico
// (the site has never shipped one — same 404 on the old GitHub Pages host).
page.on("console", (m) => {
  if (m.type() === "error" && !/googletagmanager|gtag|net::ERR|favicon/.test(m.location().url ?? "")) {
    errors.push(m.text());
  }
});
page.on("response", (r) => {
  if (r.status() >= 400 && !/favicon/.test(r.url())) {
    errors.push(`HTTP ${r.status()} ${r.url()}`);
  }
});

// ---- Home ----
await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
await page.waitForFunction(() =>
  document.getElementById("api-status")?.textContent === "connected", null, { timeout: 15000 });
check("home: api connected", true);
const stats = await (await fetch(`${BASE}/api/stats`)).json();
const rank3 = stats.by_rank.find((r) => r.n === 3).distinct_quivers;
check("home: distinct stat from /stats", await page.locator("#stat-distinct").textContent() === stats.distinct_quivers.toLocaleString("en-US"));
check("home: labeled stat from /stats", await page.locator("#stat-labeled").textContent() === stats.labeled_quivers.toLocaleString("en-US"));
check("home: featured quiver drawn",
  await page.locator("#featured-figure svg").count() === 1);

// ---- Browse ----
await page.goto(`${BASE}/browse.html`, { waitUntil: "networkidle" });
await page.evaluate((n) => { window.__rank3 = n; }, rank3);
await page.waitForFunction(() =>
  document.querySelectorAll("#table-body tr").length >= 50, null, { timeout: 15000 });
check("browse: 50 rows", true);
check("browse: api connected",
  await page.locator("#api-status").textContent() === "connected");
// Rank filter
await page.selectOption("#filter-rank", "3");
await page.evaluate(() => applyFilters());
await page.waitForFunction(() =>
  document.querySelectorAll("#table-body tr").length === window.__rank3, null, { timeout: 15000 });
check("browse: rank-3 filter shows every rank-3 quiver", true);

// ---- Search (empty until a filter is applied — set rank=3 and run) ----
await page.goto(`${BASE}/search.html`, { waitUntil: "networkidle" });
await page.evaluate((n) => { window.__rank3 = n; }, rank3);
await page.selectOption("#f-rank", "3");
await page.evaluate(() => runSearch());
await page.waitForFunction(() =>
  document.querySelectorAll("#results-area tbody tr").length === window.__rank3,
  null, { timeout: 15000 });
check("search: rank-3 search returns every rank-3 quiver", true);
check("search: count text", (await page.textContent(".results-count")).includes(String(rank3)));

// ---- Quiver detail ----
await page.goto(`${BASE}/quiver.html?id=${encodeURIComponent(qid)}`, { waitUntil: "networkidle" });
await page.waitForFunction((id) =>
  document.body.textContent.includes(id), qid, { timeout: 15000 });
check("quiver page: shows id + A3", (await page.content()).includes("A3"));
check("quiver page: draws figure", await page.locator("svg").count() >= 1);

// ---- Class detail ----
await page.goto(`${BASE}/class.html?id=${encodeURIComponent(mcId)}`, { waitUntil: "networkidle" });
await page.waitForFunction((id) =>
  document.body.textContent.includes(id), mcId, { timeout: 15000 });
check("class page: shows id", true);
check("class page: shows class size 14",
  (await page.textContent("body")).includes("14"));

// ---- Client-side xlsx from real CSV ----
await page.goto(`${BASE}/browse.html`, { waitUntil: "networkidle" });
const b64 = await page.evaluate(async () => {
  const res = await fetch("/api/export?rank=2&format=csv");
  const rows = QMDXlsx.parseCsv(await res.text());
  const blob = QMDXlsx.fromRows(rows);
  const buf = new Uint8Array(await blob.arrayBuffer());
  let s = "";
  for (const b of buf) s += String.fromCharCode(b);
  return btoa(s);
});
writeFileSync((process.env.SCRATCH ?? "/tmp") + "/qmd-test-export.xlsx", Buffer.from(b64, "base64"));
check("xlsx: generated in browser", b64.length > 100);

// ---- Download modal opens with Excel option ----
await page.click("#download-btn, [onclick*='QMDDownload']").catch(() => {});
const hasModal = await page.evaluate(() => !!window.QMDDownload && !!window.QMDXlsx);
check("download modal + xlsx lib present", hasModal);

check("no page errors", errors.length === 0, errors.slice(0, 3).join(" | "));

// ---- Error states: no fabricated data, no reflected markup ----
const payload = "<img src=x onerror=window.__pwned=1>";
await page.goto(`${BASE}/quiver.html?id=${encodeURIComponent(payload)}`, { waitUntil: "networkidle" });
check("quiver: malformed id renders a message, not a quiver",
  (await page.locator("#page-content .state-msg").count()) === 1
  && (await page.locator("#page-content .qid").count()) === 0);
check("quiver: malformed id is escaped (no script execution)",
  await page.evaluate(() => window.__pwned === undefined)
  && (await page.locator("#page-content img").count()) === 0);
await page.goto(`${BASE}/quiver.html?id=Q.n3.0000000000000000`, { waitUntil: "networkidle" });
check("quiver: unknown id -> 'No quiver with ID'",
  (await page.locator("#page-content .state-msg").textContent() ?? "").includes("No quiver with ID"));
await page.goto(`${BASE}/class.html?id=MC.n3.0000000000000000`, { waitUntil: "networkidle" });
check("class: unknown id -> 'No mutation class with ID'",
  (await page.locator("#page-content .state-msg").textContent() ?? "").includes("No mutation class with ID"));

// ---- Browse recovers from an empty result ----
await page.goto(`${BASE}/browse.html`, { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelectorAll("#table-body tr").length === 50, null, { timeout: 15000 });
await page.fill("#filter-dynkin", "Z9");
await page.evaluate(() => applyFilters());
await page.waitForFunction(() => document.querySelectorAll("#table-body tr").length === 1, null, { timeout: 15000 });
await page.evaluate(() => resetFilters());
await page.waitForFunction(() => document.querySelectorAll("#table-body tr").length === 50, null, { timeout: 15000 });
check("browse: reset after an empty result reloads the table", true);

// ---- Search deep link populates the form and runs ----
await page.goto(`${BASE}/search.html?rank=3&is_acyclic=true`, { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelectorAll("#results-area tbody tr").length > 0, null, { timeout: 15000 });
check("search: deep link fills the form",
  await page.inputValue("#f-rank") === "3" && await page.isChecked("#f-acyclic"));

// ---- Class members table: one row per distinct quiver, one canonical star ----
await page.goto(`${BASE}/quiver.html?id=Q.n4.d5a342bfb1d3d96c`, { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelectorAll("#class-members-container tbody tr").length > 0, null, { timeout: 15000 });
const memberRows = await page.locator("#class-members-container tbody tr").count();
const stars = await page.locator("#class-members-container tbody tr.is-canon").count();
const currentRows = await page.locator("#class-members-container tbody tr", { hasText: "current" }).count();
check("quiver: members table lists distinct quivers (2, not 4 labelings)", memberRows === 2, String(memberRows));
check("quiver: exactly one canonical rep", stars === 1, String(stars));
check("quiver: current quiver listed once", currentRows === 1, String(currentRows));

// ---- Class page: nickname lookup, paged members, labelings view ----
await page.goto(`${BASE}/class.html?name=markov`, { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelector("#page-content h1"), null, { timeout: 15000 });
check("class: ?name=markov resolves and titles the page", (await page.locator("#page-content h1").textContent()) === "Markov");
check("class: url rewritten to the id", page.url().includes("id=MC.n3.7405511b230b7552"));
await page.waitForFunction(() => document.querySelectorAll("#orbit-container tbody tr").length > 0, null, { timeout: 15000 });
await page.evaluate(() => setMode("all"));
await page.waitForFunction(() => document.querySelectorAll("#orbit-container tbody tr").length === 2, null, { timeout: 15000 });
check("class: 'All labelings' lists the 2 Markov labelings", true);
await page.goto(`${BASE}/class.html?id=${mcId}`, { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelectorAll("#orbit-container tbody tr").length === 4, null, { timeout: 15000 });
check("class: A3 shows 4 distinct members, canonical first",
  (await page.locator("#orbit-container tbody tr").first().locator(".canon-mark").count()) === 1);
await page.evaluate(() => setMode("all"));
await page.waitForFunction(() => document.querySelectorAll("#orbit-container tbody tr").length === 14, null, { timeout: 15000 });
check("class: A3 'All labelings' shows 14 rows", true);
await page.evaluate(() => setLayout("grid"));
await page.waitForFunction(() => document.querySelectorAll("#orbit-container .matrix-card").length === 14, null, { timeout: 15000 });
check("class: grid view renders 14 cards", true);
// Large-class path: labelings not inlined -> fetched lazily from /labelings, paged.
await page.evaluate(() => { setLayout("table"); members.all = { items: [], next: null, total: 14, endpoint: "labelings", loaded: false }; setMode("distinct"); setMode("all"); });
await page.waitForFunction(() => document.querySelectorAll("#orbit-container tbody tr").length === 14, null, { timeout: 15000 });
check("class: lazy labelings load (large-class path) fills 14 rows", true);

// ---- Browse: nickname shown, sort disabled in labelings scope ----
await page.goto(`${BASE}/browse.html?`, { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelectorAll("#table-body tr").length === 50, null, { timeout: 15000 });
await page.selectOption("#filter-rank", "3");
await page.evaluate(() => applyFilters());
await page.waitForFunction(() => document.querySelectorAll("#table-body .nick").length > 0, null, { timeout: 15000 });
check("browse: Markov nickname shown next to its class id", (await page.locator("#table-body .nick").first().textContent()) === "Markov");
await page.evaluate(() => setScope("labelings"));
await page.waitForFunction(() => document.querySelectorAll("#table-body tr").length > 1
  && document.querySelectorAll("th.sort-disabled").length > 0, null, { timeout: 15000 });
check("browse: labelings scope lists stored labelings (finite classes only), sort disabled", true);

// ---- Filter menus are generated from /stats, not hard-coded ----
// The regression these guard: rank stopped at 4 and max edge at 2 for two
// census releases, so whole ranks and every quiver of weight >= 3 were
// unreachable from the UI while sitting in the database. Assert against
// /stats rather than against a literal, or the test goes stale the same way.
const menuStats = stats;   // same payload; named apart from the home-page checks above
const classesTotal = (await (await fetch(`${BASE}/api/classes?limit=1`)).json()).total;
const ranks = menuStats.by_rank.map((r) => String(r.n));
const edgeCeiling = Math.max(0, ...menuStats.by_rank.map((r) => r.bound ?? 0));

await page.goto(`${BASE}/browse.html`, { waitUntil: "networkidle" });
await page.waitForFunction((n) => document.querySelectorAll("#filter-rank option").length === n + 1,
  ranks.length, { timeout: 15000 });
const browseRanks = await page.$$eval("#filter-rank option", (os) => os.map((o) => o.value).filter(Boolean));
check("browse: rank menu lists exactly the ranks /stats reports",
  JSON.stringify(browseRanks) === JSON.stringify(ranks), `${browseRanks} vs ${ranks}`);

await page.goto(`${BASE}/search.html`, { waitUntil: "networkidle" });
await page.waitForFunction((n) => document.querySelectorAll("#f-rank option").length === n + 1,
  ranks.length, { timeout: 15000 });
const searchRanks = await page.$$eval("#f-rank option", (os) => os.map((o) => o.value).filter(Boolean));
check("search: rank menu lists exactly the ranks /stats reports",
  JSON.stringify(searchRanks) === JSON.stringify(ranks), `${searchRanks} vs ${ranks}`);
const edges = await page.$$eval("#f-maxedge option", (os) => os.map((o) => o.value).filter((v) => v !== ""));
check("search: max-edge menu spans 0..bound from /stats",
  edges.length === edgeCeiling + 1 && edges[0] === "0" && edges[edges.length - 1] === String(edgeCeiling),
  `${edges.length} options, ceiling ${edgeCeiling}`);

// ---- Named-classes filter ----
await page.goto(`${BASE}/browse.html`, { waitUntil: "networkidle" });
await page.selectOption("#filter-named", "true");
await page.click("button.btn:has-text('Filter')");
await page.waitForFunction(() => {
  const rows = document.querySelectorAll("#table-body tr");
  return rows.length > 0 && !document.querySelector("#table-body .state-msg");
}, null, { timeout: 15000 });
const nickCells = await page.$$eval("#table-body tr", (rs) => rs.map((r) => !!r.querySelector(".nick")));
check("browse: named-classes filter returns only quivers with a nickname",
  nickCells.length > 0 && nickCells.every(Boolean), `${nickCells.filter(Boolean).length}/${nickCells.length}`);

// ---- Mutation classes view ----
await page.goto(`${BASE}/browse.html`, { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelectorAll("#table-body tr").length > 1, null, { timeout: 15000 });
await page.click("#btn-classes");
await page.waitForFunction(() => document.getElementById("table-body").dataset.scope === "classes",
  null, { timeout: 15000 });
check("browse: classes view swaps to the class header",
  await page.locator("#thead-classes").isVisible() && !(await page.locator("#thead-quivers").isVisible()));
const classLinks = await page.$$eval("#table-body tr td:first-child a",
  (as) => as.map((a) => a.textContent.trim()));
check("browse: every row is a mutation class",
  classLinks.length > 1 && classLinks.every((t) => t.startsWith("MC.n")), classLinks[0]);
check("browse: noun and counts follow the view",
  (await page.locator("#stat-noun").textContent()) === "mutation classes"
    && Number((await page.locator("#stat-showing").textContent()).replace(/[,+]/g, "")) === classesTotal,
  await page.locator("#stat-showing").textContent());
check("browse: download disabled in the classes view (CSV export is quivers-only)",
  await page.locator("#btn-download").isDisabled());

// Sorting must switch key sets: qmd_id is not a class sort and would 400.
await page.click('#thead-classes th[data-col="distinct_quiver_count"]');
// Wait for the RENDER, not merely for rows to exist: the previous table is
// still on screen and would satisfy a row-count wait immediately.
await page.waitForFunction(() => {
  const tb = document.getElementById("table-body");
  return tb.dataset.scope === "classes" && tb.dataset.sort === "distinct_quiver_count";
}, null, { timeout: 15000 });
const dqc = await page.$$eval("#table-body tr td:nth-child(6)",
  (ts) => ts.map((t) => Number(t.textContent.replace(/,/g, ""))));
check("browse: classes sort by distinct quivers ascending",
  dqc.length > 1 && dqc.every((v, i) => i === 0 || dqc[i - 1] <= v), dqc.slice(0, 5).join(","));

// ...and switching back must restore a quiver sort rather than sending a class key.
await page.click("#btn-distinct");
await page.waitForFunction(() => document.getElementById("table-body").dataset.scope === "distinct",
  null, { timeout: 15000 });
check("browse: switching back to quivers restores a valid sort",
  await page.locator("#thead-quivers").isVisible() && !(await page.locator("#thead-classes").isVisible()));

// ---- Home uses /stats and /random ----
await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
check("home: ranks covered derived from /stats",
  await page.locator("#stat-ranks").textContent() === "1–4");

await browser.close();
console.log(failures === 0 ? "\nBROWSER CHECKS PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures ? 1 : 0);
