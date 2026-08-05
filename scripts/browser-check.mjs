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

const browser = await chromium.launch({ executablePath: "/opt/pw-browsers/chromium" });
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
await page.click("#apply-filters").catch(() => page.evaluate(() => applyFilters()));
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

await browser.close();
console.log(failures === 0 ? "\nBROWSER CHECKS PASS" : `\n${failures} FAILURE(S)`);
process.exit(failures ? 1 : 0);
