/* ============================================================
   QMD — Dataset download modal (shared by browse + search)

   Usage:
     <script src="/download.js"></script>
     QMDDownload.open(filters, count)   // filters: same keys as /search

   Opens a small modal that optionally collects an email / name and lets the
   user pick a format (CSV / Excel) and contents (distinct quivers vs. all
   labelings), then hands off to the API's /export endpoint (which streams the
   file and records the download server-side).
   ============================================================ */
(function () {
  const API = '/api';

  let _filters = {};
  let _counts = { distinct: null, labelings: null };
  let _lowerBound = false;

  function activeFilters(f) {
    return Object.entries(f || {}).filter(([, v]) => v !== '' && v != null);
  }

  function selected(group) {
    const el = document.querySelector(`.dl-seg[data-group="${group}"] .seg.active`);
    return el ? el.dataset.val : null;
  }

  function buildUrl(format, scope) {
    const params = new URLSearchParams({ format, scope });
    for (const [k, v] of activeFilters(_filters)) params.set(k, v);
    const email = document.getElementById('dl-email').value.trim();
    const name  = document.getElementById('dl-name').value.trim();
    if (email) params.set('email', email);
    if (name)  params.set('name', name);
    return `${API}/export?${params.toString()}`;
  }

  function injectModal() {
    const el = document.createElement('div');
    el.className = 'dl-overlay';
    el.id = 'dl-overlay';
    el.hidden = true;
    el.innerHTML = `
      <div class="dl-modal" role="dialog" aria-modal="true" aria-labelledby="dl-title">
        <div class="dl-head">
          <h2 id="dl-title">Download dataset</h2>
          <button class="dl-close" type="button" aria-label="Close" onclick="QMDDownload.close()">&times;</button>
        </div>
        <p class="dl-sub" id="dl-sub"></p>

        <div class="dl-field">
          <label for="dl-email">Email <span class="dl-opt">(optional)</span></label>
          <input type="email" id="dl-email" placeholder="you@university.edu" autocomplete="email">
        </div>
        <div class="dl-field">
          <label for="dl-name">Name / affiliation <span class="dl-opt">(optional)</span></label>
          <input type="text" id="dl-name" placeholder="Jane Doe, Some University" autocomplete="organization">
        </div>

        <div class="dl-choice">
          <span class="dl-choice-label">Contents</span>
          <div class="dl-seg" data-group="scope">
            <button type="button" class="seg active" data-val="distinct">Distinct quivers</button>
            <button type="button" class="seg" data-val="labelings">All labelings</button>
          </div>
        </div>
        <div class="dl-choice">
          <span class="dl-choice-label">Format</span>
          <div class="dl-seg" data-group="format">
            <button type="button" class="seg active" data-val="csv">CSV</button>
            <button type="button" class="seg" data-val="xlsx">Excel</button>
          </div>
        </div>

        <p class="dl-note">Email is optional — we use it only to understand who relies on the dataset.</p>

        <!-- The filtered export above is served from the database; this is the
             whole census as static files from R2. For anything approaching the
             full dataset this is the right route: no query, resumable, and the
             checksums let you verify what landed. -->
        <details class="dl-bulk" id="dl-bulk">
          <summary>Or download the entire database</summary>
          <div id="dl-bulk-body" class="dl-bulk-body">Loading…</div>
        </details>

        <div class="dl-actions">
          <button class="btn btn-ghost" type="button" onclick="QMDDownload.close()">Cancel</button>
          <button class="btn" type="button" onclick="QMDDownload.go()">Download</button>
        </div>
      </div>`;
    document.body.appendChild(el);

    el.addEventListener('click', e => {
      const seg = e.target.closest('.seg');
      if (seg && el.contains(seg)) {
        const group = seg.parentElement;
        group.querySelectorAll('.seg').forEach(b => b.classList.remove('active'));
        seg.classList.add('active');
        if (group.dataset.group === 'scope') updateSub();
        return;
      }
      if (e.target === el) close();
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && !el.hidden) close();
    });
  }

  function updateSub() {
    const n = activeFilters(_filters).length;
    const scope = selected('scope') || 'distinct';
    const c = _counts[scope];
    const noun = scope === 'labelings' ? 'labeled quiver' : 'quiver';
    const cntTxt = (c == null)
      ? (scope === 'labelings' ? 'labeled quivers' : 'quivers')
      : `<b>${Number(c).toLocaleString()}${_lowerBound ? '+' : ''}</b> ${noun}${c === 1 && !_lowerBound ? '' : 's'}`;
    const cut = n === 0 ? 'the full dataset (no filters)' : 'your current filter cut';
    document.getElementById('dl-sub').innerHTML = `Exporting ${cntTxt} — ${cut}.`;
  }

  // counts: a number (same for both) or { distinct, labelings, lowerBound }.
  // lowerBound marks a count the API capped (total_is_lower_bound); the export
  // itself is unaffected -- it streams the whole cut either way -- so this only
  // changes "12,345 quivers" into "10,000+ quivers" rather than overstating it.
  function open(filters, counts, initialScope) {
    _filters = filters || {};
    _lowerBound = false;
    if (counts == null) {
      _counts = { distinct: null, labelings: null };
    } else if (typeof counts === 'number') {
      _counts = { distinct: counts, labelings: counts };
    } else {
      _counts = {
        distinct:  counts.distinct  == null ? null : counts.distinct,
        labelings: counts.labelings == null ? null : counts.labelings,
      };
      _lowerBound = !!counts.lowerBound;
    }
    const scope = initialScope === 'labelings' ? 'labelings' : 'distinct';
    document.querySelectorAll('.dl-seg[data-group="scope"] .seg').forEach(b =>
      b.classList.toggle('active', b.dataset.val === scope));
    updateSub();
    document.getElementById('dl-overlay').hidden = false;
    const bulk = document.getElementById('dl-bulk');
    if (bulk && !bulk.dataset.wired) {
      bulk.dataset.wired = '1';
      bulk.addEventListener('toggle', () => { if (bulk.open) loadBulk(); });
    }
    document.getElementById('dl-email').focus();
  }

  // Fetched once per page, on first open of the section: the manifest is ~5 KB
  // and changes only at a data release.
  let _bulkLoaded = false;
  async function loadBulk() {
    if (_bulkLoaded) return;
    _bulkLoaded = true;
    const body = document.getElementById('dl-bulk-body');
    try {
      const r = await fetch('/api/bulk');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const m = await r.json();
      const gb = (m.total_bytes_gz / 1e9).toFixed(2);
      const rows = m.ranks.map(p => `
        <tr>
          <td class="mono">${p.rank}</td>
          <td class="mono">${Number(p.rows).toLocaleString()}</td>
          <td class="mono">${(p.bytes_gz / 1e6).toFixed(1)} MB</td>
          <td><a href="/api/bulk/${p.file}">${p.file}</a></td>
        </tr>`).join('');
      body.innerHTML = `
        <p>The complete census as gzipped NDJSON, one file per rank —
           <b>${Number(m.total_rows).toLocaleString()}</b> rows, ${gb} GB compressed.
           Rows are exactly what <code>/api/export.ndjson</code> serves.</p>
        <div class="table-wrap">
          <table class="dl-bulk-table">
            <thead><tr><th>Rank</th><th>Rows</th><th>Size</th><th>File</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
        <p class="dl-note">
          Downloads resume (<code>curl -C - -O</code>). Verify with
          <code>gunzip -c FILE | shasum -a 256</code> against
          <code>sha256_ndjson</code> in
          <a href="/api/bulk/manifest.json">manifest.json</a>.
          Also: <a href="/api/bulk/rank_stats.json">rank_stats.json</a> ·
          <a href="/api/bulk/nicknames.json">nicknames.json</a>.
          Licensed CC-BY-4.0 — please cite.
        </p>`;
    } catch (e) {
      console.error(e);
      body.innerHTML = `<p class="dl-note">The bulk corpus is unavailable right now.
        Use <code>/api/export.ndjson</code> (resumable via the
        <code>X-Next-Cursor</code> header) instead.</p>`;
    }
  }

  function close() {
    const el = document.getElementById('dl-overlay');
    if (el) el.hidden = true;
  }

  function go() {
    const format = selected('format') || 'csv';
    const scope  = selected('scope')  || 'distinct';
    if (format === 'xlsx') {
      // The API serves CSV only; Excel is built client-side from it
      // (xlsx-lite.js), so the Worker never generates xlsx.
      goXlsx(scope);
    } else {
      // Same-origin attachment: the server's Content-Disposition supplies
      // the filename, so a plain link click downloads without navigating.
      const a = document.createElement('a');
      a.href = buildUrl('csv', scope);
      a.rel = 'noopener';
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
    if (window.gtag) {
      gtag('event', 'download', {
        format,
        scope,
        filtered: activeFilters(_filters).length > 0,
      });
    }
    close();
  }

  async function goXlsx(scope) {
    try {
      const res = await fetch(buildUrl('csv', scope));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Keep the server's timestamped filename, just with the xlsx extension.
      const disp = res.headers.get('content-disposition') || '';
      const m = /filename="?([^";]+)\.csv"?/.exec(disp);
      const filename = (m ? m[1] : `qmd-${scope === 'labelings' ? 'labelings' : 'quivers'}`) + '.xlsx';
      const rows = QMDXlsx.parseCsv(await res.text());
      const url = URL.createObjectURL(QMDXlsx.fromRows(rows));
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 30000);
    } catch (e) {
      alert('Excel export failed — please try the CSV format.\n(' + e + ')');
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectModal);
  } else {
    injectModal();
  }

  window.QMDDownload = { open, close, go };
})();
