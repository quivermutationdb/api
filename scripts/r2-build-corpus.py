#!/usr/bin/env python3
"""
Build the R2 bulk corpus: one gzipped NDJSON file per rank, plus a manifest.

Rows are byte-for-byte the shape /api/export.ndjson serves (EXPORT_COLUMNS in
src/api/export.ts, exportRow for the values), so the corpus and the API agree
and a user can move between them without reconciling two schemas. That is not
free -- it means decoding the compact upper-triangular matrix here exactly as
src/db/matrix.ts does -- but a bulk file that disagrees with the API is worse
than no bulk file.

Source is a LOCAL Postgres holding the census, not the production database:
50.8 M rows is ~12 GB of egress otherwise, and the local copy can be verified
identical first (see docs/PLANETSCALE.md). It must have every release patch
applied -- drizzle-pg/0003 and the curated nicknames -- or the corpus will
disagree with what the site serves.

  psql -d qmd_footprint -f drizzle-pg/0003_rank8_dangling_class_refs.sql
  python scripts/r2-build-corpus.py dist/r2 --database qmd_footprint

Streams through `psql --command COPY ... TO STDOUT`: no driver dependency, and
constant memory regardless of rank size. Values arrive as COPY text, which
escapes backslash/tab/newline/CR -- unescaped here before use. JSON is built
with json.dumps so the output escaping is correct by construction; emitting
JSON straight from Postgres via row_to_json would double-escape every
backslash inside a string.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qmd.encoding import decode_upper  # noqa: E402

# Column order is the API's, and appending-only is a compatibility promise.
COLUMNS = [
    "qmd_id", "num_vertices", "exchange_matrix", "representation_type",
    "max_edge", "is_acyclic", "is_connected", "is_bipartite", "is_abundant",
    "is_planar", "symmetry_order", "symmetry_name",
    "mc_id", "dynkin_type", "is_open", "class_size", "labeled_size",
    "distinct_quiver_count", "merged_orbit_count",
    "is_finite_confirmed", "is_infinite_confirmed", "is_infinite_expected",
    "size_of_explored_frontier", "is_mutation_acyclic",
    "is_banff", "is_louise", "is_p_prime",
    "exploration", "nickname", "mutation_finite", "explored",
]

# (n, seq) is id order per rank, which is the order /api/export.ndjson walks.
QUERY = """
COPY (
  SELECT q.id, q.n, q.exchange_matrix, q.representation_type, q.max_edge,
         q.is_acyclic, q.is_connected, q.is_bipartite, q.is_abundant, q.is_planar,
         q.symmetry_group ->> 'order', q.symmetry_group ->> 'name',
         q.mutation_class_id, mc.dynkin_type, mc.is_open, mc.class_size,
         mc.distinct_quiver_count, mc.merged_orbit_count,
         mc.is_finite_confirmed, mc.is_infinite_confirmed, mc.is_infinite_expected,
         mc.size_of_explored_frontier, mc.is_mutation_acyclic,
         mc.is_banff, mc.is_louise, mc.is_p_prime, mc.exploration,
         nk.nickname, q.mutation_finite
  FROM quivers q
  LEFT JOIN mutation_classes mc ON q.mutation_class_id = mc.id
  LEFT JOIN class_nicknames nk ON nk.mc_id = mc.id
  WHERE q.n = {n}
  ORDER BY q.n, q.seq
) TO STDOUT
"""


def unescape(v: str):
    """COPY text -> Python value. \\N is NULL; the four escapes are literal."""
    if v == r"\N":
        return None
    if "\\" not in v:
        return v
    out, i = [], 0
    while i < len(v):
        c = v[i]
        if c == "\\" and i + 1 < len(v):
            nxt = v[i + 1]
            out.append({"t": "\t", "n": "\n", "r": "\r", "\\": "\\"}.get(nxt, nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def tri(v):
    """Postgres boolean -> Python tri-state. NULL stays None (unknown)."""
    return None if v is None else (v == "t")


def num(v):
    return None if v is None else int(v)


def row_to_obj(f: list) -> dict:
    n = int(f[1])
    return {
        "qmd_id": f[0],
        "num_vertices": n,
        # The API ships the matrix as a JSON *string*, not a nested array.
        # Matching it keeps the two outputs interchangeable.
        "exchange_matrix": json.dumps(decode_upper(n, f[2] or ""), separators=(",", ":")),
        "representation_type": f[3],
        "max_edge": num(f[4]),
        "is_acyclic": tri(f[5]),
        "is_connected": tri(f[6]),
        "is_bipartite": tri(f[7]),
        "is_abundant": tri(f[8]),
        "is_planar": tri(f[9]),
        "symmetry_order": num(f[10]),
        "symmetry_name": f[11],
        "mc_id": f[12],
        "dynkin_type": f[13],
        # is_open is false, not null, when there is no class row -- as the API does.
        "is_open": tri(f[14]) if f[14] is not None else False,
        # class_size is the labeled orbit only when the class is CLOSED; an open
        # class has no finite size, and reporting the explored count there would
        # read as an exact answer to an unanswered question.
        "class_size": num(f[15]) if tri(f[14]) is False else None,
        "labeled_size": num(f[15]),
        "distinct_quiver_count": num(f[16]),
        "merged_orbit_count": num(f[17]),
        "is_finite_confirmed": tri(f[18]),
        "is_infinite_confirmed": tri(f[19]),
        "is_infinite_expected": tri(f[20]),
        "size_of_explored_frontier": num(f[21]),
        "is_mutation_acyclic": tri(f[22]),
        "is_banff": tri(f[23]),
        "is_louise": tri(f[24]),
        "is_p_prime": tri(f[25]),
        "exploration": f[26] if f[12] is not None else None,
        "nickname": f[27],
        "mutation_finite": tri(f[28]),
        "explored": f[12] is not None,
    }


def build_rank(n: int, out_dir: str, database: str, host: str) -> dict:
    path = os.path.join(out_dir, f"qmd-n{n}.ndjson.gz")
    # -q, or psql echoes "SET" onto stdout as a line with no tab in it and the
    # first row parses as garbage. statement_timeout = 0 because the role
    # carries a 10 s cap for the API's protection; a full-rank COPY is exactly
    # the long-running query that cap exists to stop.
    cmd = ["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-d", database,
           "-c", "SET statement_timeout = 0",
           "-c", QUERY.format(n=n)]
    if host:
        cmd[1:1] = ["-h", host]
    rows = 0
    sha = hashlib.sha256()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20) as proc, \
            open(path, "wb") as raw:
        # mtime=0: the gzip header otherwise embeds a timestamp, so the same
        # data would produce a different sha256 on every run and the manifest
        # could not be used to tell "unchanged" from "regenerated".
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as gz:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                fields = [unescape(v) for v in line.split("\t")]
                blob = (json.dumps(row_to_obj(fields), separators=(",", ":"),
                                   ensure_ascii=False) + "\n").encode("utf-8")
                gz.write(blob)
                sha.update(blob)
                rows += 1
                if rows % 1_000_000 == 0:
                    print(f"    rank {n}: {rows:,}", flush=True)
        if proc.wait() != 0:
            raise SystemExit(f"psql failed for rank {n}")
    size = os.path.getsize(path)
    print(f"  rank {n}: {rows:,} rows -> {os.path.basename(path)} "
          f"({size / 1e6:.1f} MB gz)", flush=True)
    return {
        "file": os.path.basename(path),
        "rank": n,
        "rows": rows,
        "bytes_gz": size,
        # sha256 of the UNCOMPRESSED ndjson: gzip output can vary with zlib
        # version, the content cannot, so this is what a user can verify.
        "sha256_ndjson": sha.hexdigest(),
        "sha256_gz": hashlib.sha256(open(path, "rb").read()).hexdigest()
        if size < 500_000_000 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--database", default="qmd_footprint")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ranks", default=None, help="comma-separated (default: all present)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    def q(sql: str) -> str:
        cmd = ["psql", "-X", "-tA", "-d", a.database, "-c", sql]
        if a.host:
            cmd[1:1] = ["-h", a.host]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()

    ranks = [int(r) for r in a.ranks.split(",")] if a.ranks else \
        [int(r) for r in q("SELECT n FROM quivers GROUP BY n ORDER BY n").split("\n") if r]
    print(f"building corpus for ranks {ranks} from {a.database}")

    parts = [build_rank(n, a.out_dir, a.database, a.host) for n in ranks]

    # Small tables ship whole: they are the context the per-rank files lack.
    aux = {}
    for table, sql in (
        ("rank_stats", "SELECT json_agg(row_to_json(t) ORDER BY t.n) FROM rank_stats t"),
        ("nicknames", "SELECT json_agg(row_to_json(t) ORDER BY t.slug) FROM class_nicknames t"),
    ):
        path = os.path.join(a.out_dir, f"{table}.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(q(sql) + "\n")
        aux[table] = {"file": f"{table}.json",
                      "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
        print(f"  {table} -> {table}.json")

    manifest = {
        "dataset": "Quiver Mutation Database — full census",
        "licence": "CC-BY-4.0",
        "cite": "Blake Jackson, Quiver Mutation Database, https://quivermutationdb.org",
        "homepage": "https://quivermutationdb.org",
        "format": "gzipped NDJSON, one JSON object per line",
        "row_shape": "identical to GET /api/export.ndjson (EXPORT_COLUMNS)",
        "columns": COLUMNS,
        "order": "rank ascending, then id order within a rank",
        "total_rows": sum(p["rows"] for p in parts),
        "total_bytes_gz": sum(p["bytes_gz"] for p in parts),
        "ranks": parts,
        "aux": aux,
        "notes": [
            "Semidecidable properties are three-state: true / false / null = "
            "unknown. Never read null as 'no'.",
            "exchange_matrix is a JSON string holding a row-major "
            "skew-symmetric matrix; b_ij > 0 means b_ij arrows i -> j.",
            "The census holds connected quivers only.",
            "sha256_ndjson is of the UNCOMPRESSED content: gunzip and hash to "
            "verify, since gzip bytes can differ between zlib versions.",
        ],
    }
    with open(os.path.join(a.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"  -> manifest.json  ({manifest['total_rows']:,} rows, "
          f"{manifest['total_bytes_gz'] / 1e9:.2f} GB gz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
