/**
 * Lex-min canonical form of a skew-symmetric matrix — a port of
 * qmd/canonicalize.lexmin_form (branch and bound with twin-vertex pruning),
 * so the Worker can turn a pasted matrix into its Q.* id. Identical results
 * to the Python definition (the ID key is lex-min by definition).
 */

import type { Matrix } from "./db/schema";

export function isSkewSymmetric(m: Matrix): boolean {
  const n = m.length;
  if (!m.every((r) => r.length === n)) return false;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (!Number.isInteger(m[i]![j]!) || m[i]![j]! !== -m[j]![i]!) return false;
    }
  }
  return true;
}

export function isConnected(m: Matrix): boolean {
  const n = m.length;
  if (n === 0) return false;
  const seen = new Set<number>([0]);
  const stack = [0];
  while (stack.length) {
    const v = stack.pop()!;
    for (let w = 0; w < n; w++) {
      if (m[v]![w] !== 0 && !seen.has(w)) { seen.add(w); stack.push(w); }
    }
  }
  return seen.size === n;
}

function apply(m: Matrix, perm: number[]): Matrix {
  return perm.map((pi) => perm.map((pj) => m[pi]![pj]!));
}

function lexKey(m: Matrix): number[] {
  return m.flat();
}

function lexLess(a: number[], b: number[], len: number): number {
  for (let i = 0; i < len; i++) {
    if (a[i]! !== b[i]!) return a[i]! < b[i]! ? -1 : 1;
  }
  return 0;
}

export function lexminForm(m: Matrix): Matrix {
  const n = m.length;
  if (n <= 1) return m.map((r) => [...r]);

  // Twin classes: (u v) is an automorphism iff b_uv = 0 and identical rows elsewhere.
  const twin = Array.from({ length: n }, (_, i) => i);
  const find = (x: number): number => { while (twin[x] !== x) { twin[x] = twin[twin[x]!]!; x = twin[x]!; } return x; };
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      if (m[u]![v] !== 0) continue;
      let same = true;
      for (let w = 0; w < n && same; w++) if (w !== u && w !== v && m[u]![w] !== m[v]![w]) same = false;
      if (same) twin[find(v)] = find(u);
    }
  }

  let bestKey = lexKey(m);
  let bestPerm = Array.from({ length: n }, (_, i) => i);

  const dfs = (prefix: number[], remaining: number[]): void => {
    const k = prefix.length;
    if (k === n) {
      const key = lexKey(apply(m, prefix));
      if (lexLess(key, bestKey, key.length) < 0) { bestKey = key; bestPerm = [...prefix]; }
      return;
    }
    const bound: number[] = [];
    for (const r of prefix) {
      const row = m[r]!;
      for (const c of prefix) bound.push(row[c]!);
      bound.push(...remaining.map((v) => row[v]!).sort((a, b) => a - b));
    }
    if (lexLess(bound, bestKey, bound.length) > 0) return;
    const seen = new Set<number>();
    const cands: number[] = [];
    for (const v of [...remaining].sort((a, b) => (m[prefix[0]!]![a]! - m[prefix[0]!]![b]!) || (a - b))) {
      const r = find(v);
      if (seen.has(r)) continue;
      seen.add(r);
      cands.push(v);
    }
    for (const v of cands) {
      prefix.push(v);
      dfs(prefix, remaining.filter((u) => u !== v));
      prefix.pop();
    }
  };
  const seenStart = new Set<number>();
  for (let s = 0; s < n; s++) {
    const r = find(s);
    if (seenStart.has(r)) continue;
    seenStart.add(r);
    dfs([s], Array.from({ length: n }, (_, i) => i).filter((v) => v !== s));
  }
  return apply(m, bestPerm);
}

/** Q.n{rank}.{sha256[:16]} of the lex-min form (JSON, compact separators, as Python does). */
export async function quiverId(canonical: Matrix): Promise<string> {
  const blob = JSON.stringify(canonical);          // no spaces == Python separators=(',', ':')
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(blob));
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  return `Q.n${canonical.length}.${hex.slice(0, 16)}`;
}
