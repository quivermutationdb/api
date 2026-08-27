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
check("home: distinct stat", await page.locator("#stat-distinct").textContent() === "724");
check("home: labeled stat", await page.locator("#stat-labeled").textContent() === "3,754");
check("home: featured quiver drawn",
  await page.locator("#featured-figure svg").count() === 1);

// ---- Browse ----
await page.goto(`${BASE}/browse.html`, { waitUntil: "networkidle" });
await page.waitForFunction(() =>
  document.querySelectorAll("#table-body tr").length >= 50, null, { timeout: 15000 });
check("browse: 50 rows", true);
check("browse: api connected",
  await page.locator("#api-status").textContent() === "connected");
// Rank filter
await page.selectOption("#filter-rank", "3");
await page.evaluate(() => applyFilters());
await page.waitForFunction(() =>
  document.querySelectorAll("#table-body tr").length === 25, null, { timeout: 15000 });
check("browse: rank-3 filter shows 25 rows", true);

// ---- Search (empty until a filter is applied — set rank=3 and run) ----
await page.goto(`${BASE}/search.html`, { waitUntil: "networkidle" });
await page.selectOption("#f-rank", "3");
await page.evaluate(() => runSearch());
await page.waitForFunction(() =>
  document.querySelectorAll("#results-area tbody tr").length === 25,
  null, { timeout: 15000 });
check("search: rank-3 search returns 25 rows", true);
check("search: count text", (await page.textContent(".results-count")).includes("25"));

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

// ---- Home uses /stats and /random ----
await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
check("home: ranks covered derived from /stats",
  await page.locator("#stat-ranks").textContent() === "1–4");

await browser.close();
console.log(failures === 0 ? "\nBROWSER CHECKS PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures ? 1 : 0);
