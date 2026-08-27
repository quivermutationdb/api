"""
qmd/encoding.py — compact storage form of an exchange matrix.

A skew-symmetric matrix is determined by its strictly-upper-triangular entries,
so the database stores only those, row by row, as a comma-separated string:

    [[0,1,-2],[-1,0,3],[2,-3,0]]  ->  "1,-2,3"

(n = 3 entries for n = 3; n(n-1)/2 in general). Roughly a third of the bytes
of the JSON form at rank 6 and less at rank 8. The rank is always known from
the id prefix or the row's `n`, so decoding is unambiguous. The Worker has the
same pair of functions in src/db/matrix.ts.
"""

from __future__ import annotations

from qmd.canonicalize import Matrix


def encode_upper(matrix: Matrix) -> str:
    n = len(matrix)
    return ",".join(str(matrix[i][j]) for i in range(n) for j in range(i + 1, n))


def decode_upper(n: int, text: str) -> Matrix:
    vals = [int(x) for x in text.split(",")] if text else []
    if len(vals) != n * (n - 1) // 2:
        raise ValueError(f"expected {n * (n - 1) // 2} entries for n={n}, got {len(vals)}")
    rows = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            rows[i][j] = vals[k]
            rows[j][i] = -vals[k]
            k += 1
    return tuple(tuple(r) for r in rows)
