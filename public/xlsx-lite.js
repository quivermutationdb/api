/* ============================================================
   QMD — minimal client-side XLSX writer (no dependencies)

   The API serves CSV only; Excel downloads are produced in the browser by
   converting that CSV to a real .xlsx (a zip of OOXML parts, stored
   uncompressed). Mirrors the old server-side openpyxl output: one "quivers"
   sheet, bold frozen header row, numbers as numbers, everything else as
   inline strings.

   Usage:
     const rows = QMDXlsx.parseCsv(csvText);      // [[cell, ...], ...]
     const blob = QMDXlsx.fromRows(rows);         // Blob (.xlsx)
   ============================================================ */
(function () {
  'use strict';

  // ---- CSV (RFC 4180: quoted fields, doubled quotes, CRLF) ----
  function parseCsv(text) {
    if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);   // strip BOM
    const rows = [];
    let row = [], field = '', inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQuotes) {
        if (ch === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; }
          else inQuotes = false;
        } else field += ch;
      } else if (ch === '"') {
        inQuotes = true;
      } else if (ch === ',') {
        row.push(field); field = '';
      } else if (ch === '\n' || ch === '\r') {
        if (ch === '\r' && text[i + 1] === '\n') i++;
        row.push(field); field = '';
        rows.push(row); row = [];
      } else {
        field += ch;
      }
    }
    if (field !== '' || row.length) { row.push(field); rows.push(row); }
    return rows;
  }

  // ---- XML helpers ----
  const xml = s => String(s)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;');

  function colName(i) {                    // 0 -> A, 25 -> Z, 26 -> AA
    let s = '';
    for (i += 1; i > 0; i = Math.floor((i - 1) / 26)) {
      s = String.fromCharCode(65 + ((i - 1) % 26)) + s;
    }
    return s;
  }

  const NUMERIC = /^-?\d+(\.\d+)?$/;

  function sheetXml(rows) {
    let out = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      + '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
      + '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
      + 'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
      + '<sheetData>';
    rows.forEach((cells, r) => {
      out += `<row r="${r + 1}">`;
      cells.forEach((v, c) => {
        const ref = colName(c) + (r + 1);
        const style = r === 0 ? ' s="1"' : '';          // bold header
        if (v === '') return;                            // empty cell: omit
        if (r > 0 && NUMERIC.test(v)) {
          out += `<c r="${ref}"${style}><v>${v}</v></c>`;
        } else {
          out += `<c r="${ref}"${style} t="inlineStr"><is><t xml:space="preserve">${xml(v)}</t></is></c>`;
        }
      });
      out += '</row>';
    });
    return out + '</sheetData></worksheet>';
  }

  const CONTENT_TYPES = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    + '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    + '<Default Extension="xml" ContentType="application/xml"/>'
    + '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    + '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
    + '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
    + '</Types>';

  const ROOT_RELS = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    + '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
    + '</Relationships>';

  const WORKBOOK = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
    + 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    + '<sheets><sheet name="quivers" sheetId="1" r:id="rId1"/></sheets></workbook>';

  const WORKBOOK_RELS = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    + '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
    + '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    + '</Relationships>';

  const STYLES = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    + '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
    + '<font><sz val="11"/><name val="Calibri"/><b/></font></fonts>'
    + '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
    + '<borders count="1"><border/></borders>'
    + '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
    + '<cellXfs count="2"><xf xfId="0"/><xf xfId="0" fontId="1" applyFont="1"/></cellXfs>'
    + '</styleSheet>';

  // ---- zip (store only, no compression) ----
  const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();

  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) {
      c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    }
    return (c ^ 0xffffffff) >>> 0;
  }

  function zipStore(entries) {
    const enc = new TextEncoder();
    const chunks = [], central = [];
    let offset = 0;
    const u16 = v => new Uint8Array([v & 0xff, (v >> 8) & 0xff]);
    const u32 = v => new Uint8Array([v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >>> 24) & 0xff]);

    for (const { name, text } of entries) {
      const nameB = enc.encode(name);
      const data = enc.encode(text);
      const crc = crc32(data);
      const local = [u32(0x04034b50), u16(20), u16(0), u16(0), u16(0), u16(0),
        u32(crc), u32(data.length), u32(data.length), u16(nameB.length), u16(0)];
      chunks.push(...local, nameB, data);
      central.push([u32(0x02014b50), u16(20), u16(20), u16(0), u16(0), u16(0), u16(0),
        u32(crc), u32(data.length), u32(data.length), u16(nameB.length),
        u16(0), u16(0), u16(0), u16(0), u32(0), u32(offset)], nameB);
      offset += local.reduce((a, b) => a + b.length, 0) + nameB.length + data.length;
    }
    const centralFlat = central.flat();
    const centralSize = centralFlat.reduce((a, b) => a + b.length, 0);
    chunks.push(...centralFlat,
      u32(0x06054b50), u16(0), u16(0), u16(entries.length), u16(entries.length),
      u32(centralSize), u32(offset), u16(0));
    return new Blob(chunks, {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    });
  }

  function fromRows(rows) {
    return zipStore([
      { name: '[Content_Types].xml', text: CONTENT_TYPES },
      { name: '_rels/.rels', text: ROOT_RELS },
      { name: 'xl/workbook.xml', text: WORKBOOK },
      { name: 'xl/_rels/workbook.xml.rels', text: WORKBOOK_RELS },
      { name: 'xl/styles.xml', text: STYLES },
      { name: 'xl/worksheets/sheet1.xml', text: sheetXml(rows) },
    ]);
  }

  window.QMDXlsx = { parseCsv, fromRows };
})();
