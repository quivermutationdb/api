/* ============================================================
   QMD — Quiver drawing
   Renders a quiver (directed multigraph) from a skew-symmetric
   exchange matrix B as an inline SVG element.

   Convention: for i < j, B[i][j] > 0 means |B[i][j]| arrows i -> j;
   B[i][j] < 0 means |B[i][j]| arrows j -> i.  Multiplicity (weight)
   is shown as a numeric label rather than parallel arrows, so it
   scales to the larger edge weights planned for the database.

   Vertices are placed on a circle, then given a small deterministic
   jitter (seeded by the matrix, so a given quiver always renders the
   same).  Several jittered candidates are scored for readability and
   the best is kept; the un-jittered circle is one of the candidates,
   so already-clean quivers stay tidy.

   Usage:  container.appendChild(renderQuiver(matrix));
   ============================================================ */
(function (global) {
  const SVGNS = 'http://www.w3.org/2000/svg';
  let _uid = 0;

  function el(name, attrs) {
    const e = document.createElementNS(SVGNS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  // --- deterministic PRNG (mulberry32) seeded from the matrix --------------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function seedFrom(matrix) {
    let h = 2166136261 >>> 0;                      // FNV-1a over the entries
    for (const row of matrix) for (const v of row) h = Math.imul(h ^ ((v | 0) >>> 0), 16777619);
    return h >>> 0;
  }

  // --- geometry helpers -----------------------------------------------------
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

  function segPtDist(p, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const L2 = dx * dx + dy * dy;
    if (L2 === 0) return dist(p, a);
    let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2;
    t = Math.max(0, Math.min(1, t));
    return dist(p, [a[0] + t * dx, a[1] + t * dy]);
  }

  // candidate 0 => no jitter; otherwise small seeded jitter on angle + radius
  function layout(matrix, cand, geo) {
    const { n, cx, cy, R } = geo;
    if (n === 1) return [[cx, cy]];
    const rnd = mulberry32((seedFrom(matrix) ^ Math.imul(cand, 0x9E3779B1)) >>> 0);
    const jit = cand > 0;
    const pos = [];
    for (let i = 0; i < n; i++) {
      let a = -Math.PI / 2 + (2 * Math.PI * i) / n;
      let rr = R;
      if (jit) {
        a += (rnd() * 2 - 1) * 0.45 * (Math.PI / n);   // up to ~half a slot
        rr = R * (1 + (rnd() * 2 - 1) * 0.15);          // +-15% (stays in view)
      }
      pos.push([cx + rr * Math.cos(a), cy + rr * Math.sin(a)]);
    }
    return pos;
  }

  // anchor point of each weight label (must match drawEdge's formula)
  function labelPoints(matrix, pos, geo) {
    const { n, r, gap, off } = geo;
    const out = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const w = matrix[i][j];
        if (!w || Math.abs(w) <= 1) continue;
        const [s, t] = w > 0 ? [i, j] : [j, i];
        const P = pos[s], Q = pos[t];
        const dx = Q[0] - P[0], dy = Q[1] - P[1], L = Math.hypot(dx, dy) || 1;
        const ux = dx / L, uy = dy / L;
        const x1 = P[0] + ux * r, y1 = P[1] + uy * r;
        const x2 = Q[0] - ux * (r + gap), y2 = Q[1] - uy * (r + gap);
        out.push({ i, j, p: [(x1 + x2) / 2 - uy * off, (y1 + y2) / 2 + ux * off] });
      }
    }
    return out;
  }

  // readability score: maximise the smallest gap between labels, vertices and
  // the edges they don't belong to
  function scoreLayout(matrix, pos, geo) {
    const { n } = geo;
    const L = labelPoints(matrix, pos, geo);
    const segs = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (matrix[i][j]) segs.push([i, j, pos[i], pos[j]]);
    let m = Infinity;
    for (let a = 0; a < L.length; a++) {
      for (let b = a + 1; b < L.length; b++) m = Math.min(m, dist(L[a].p, L[b].p));
      for (let v = 0; v < pos.length; v++) m = Math.min(m, dist(L[a].p, pos[v]));
      for (const [i, j, P, Q] of segs) {
        if (i === L[a].i && j === L[a].j) continue;
        m = Math.min(m, segPtDist(L[a].p, P, Q));
      }
    }
    for (let a = 0; a < pos.length; a++)
      for (let b = a + 1; b < pos.length; b++) m = Math.min(m, dist(pos[a], pos[b]));
    return m;
  }

  function bestLayout(matrix, geo) {
    if (geo.n < 3) return layout(matrix, 0, geo);
    let best = null;
    for (let c = 0; c < 28; c++) {
      const pos = layout(matrix, c, geo);
      const sc = scoreLayout(matrix, pos, geo);
      if (!best || sc > best.sc) best = { sc, pos };
    }
    return best.pos;
  }

  // --- drawing --------------------------------------------------------------
  function renderQuiver(matrix, opts) {
    opts = opts || {};
    const n = matrix.length;
    const mini = !!opts.mini;
    const size = opts.size || (mini ? 132 : 300);
    const r = opts.nodeRadius || (mini ? 9 : 18);
    const pad = r + (mini ? 9 : 24);
    const geo = {
      n, size, mini, r,
      cx: size / 2, cy: size / 2,
      R: Math.max(0, size / 2 - pad),
      gap: mini ? 2 : 3,
      off: mini ? 7 : 11,
    };

    const svg = el('svg', {
      viewBox: `0 0 ${size} ${size}`,
      class: 'quiver-svg' + (mini ? ' quiver-mini' : ''),
      width: size, height: size, role: 'img',
      'aria-label': `Quiver on ${n} vertices`,
    });

    const mid = 'qa-' + (_uid++);
    const ah = mini ? 6 : 9;
    const defs = el('defs', {});
    const marker = el('marker', {
      id: mid, markerWidth: ah, markerHeight: ah,
      refX: ah - 1, refY: ah / 2, orient: 'auto', markerUnits: 'userSpaceOnUse',
    });
    marker.appendChild(el('path', { d: `M0,0 L${ah},${ah / 2} L0,${ah} z`, class: 'quiver-arrowhead' }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    const pos = bestLayout(matrix, geo);

    const eg = svg.appendChild(el('g', {}));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const w = matrix[i][j];
        if (!w) continue;
        const [s, t] = w > 0 ? [i, j] : [j, i];
        drawEdge(eg, pos[s], pos[t], geo, Math.abs(w), mid);
      }
    }

    const ng = svg.appendChild(el('g', {}));
    for (let i = 0; i < n; i++) {
      const [x, y] = pos[i];
      ng.appendChild(el('circle', { cx: x, cy: y, r: r, class: 'quiver-node' }));
      const label = el('text', {
        x: x, y: y, class: 'quiver-node-label',
        'font-size': mini ? 9 : 14,
        'text-anchor': 'middle', 'dominant-baseline': 'central',
      });
      label.textContent = i + 1;
      ng.appendChild(label);
    }
    return svg;
  }

  function drawEdge(parent, P, Q, geo, weight, markerId) {
    const { r, gap, off, mini } = geo;
    const dx = Q[0] - P[0], dy = Q[1] - P[1];
    const L = Math.hypot(dx, dy) || 1;
    const ux = dx / L, uy = dy / L;
    const x1 = P[0] + ux * r, y1 = P[1] + uy * r;
    const x2 = Q[0] - ux * (r + gap), y2 = Q[1] - uy * (r + gap);
    parent.appendChild(el('line', {
      x1, y1, x2, y2, class: 'quiver-edge', 'marker-end': `url(#${markerId})`,
    }));
    if (weight > 1) {
      const lx = (x1 + x2) / 2 - uy * off, ly = (y1 + y2) / 2 + ux * off;
      const t = el('text', {
        x: lx, y: ly, class: 'quiver-weight',
        'font-size': mini ? 8 : 12,
        'text-anchor': 'middle', 'dominant-baseline': 'central',
      });
      t.textContent = weight;
      parent.appendChild(t);
    }
  }

  global.renderQuiver = renderQuiver;
})(window);
