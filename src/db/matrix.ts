/**
 * Compact storage form of an exchange matrix (mirrors qmd/encoding.py):
 * the strictly-upper-triangular entries, row by row, comma-separated.
 *   [[0,1,-2],[-1,0,3],[2,-3,0]]  <->  "1,-2,3"
 */

import type { Matrix } from "./schema";

export function encodeUpper(m: Matrix): string {
  const out: number[] = [];
  for (let i = 0; i < m.length; i++) for (let j = i + 1; j < m.length; j++) out.push(m[i]![j]!);
  return out.join(",");
}

export function decodeUpper(n: number, text: string): Matrix {
  const vals = text === "" ? [] : text.split(",").map(Number);
  if (vals.length !== (n * (n - 1)) / 2 || vals.some((v) => !Number.isInteger(v))) {
    throw new Error(`bad upper-triangular encoding for n=${n}`);
  }
  const rows: Matrix = Array.from({ length: n }, () => Array(n).fill(0));
  let k = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      rows[i]![j] = vals[k]!;
      rows[j]![i] = -vals[k]!;
      k++;
    }
  }
  return rows;
}
