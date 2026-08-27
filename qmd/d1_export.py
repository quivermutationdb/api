"""
qmd/d1_export.py

GenerationResult -> per-rank SQL for Cloudflare D1 (schema v2, docs/PHASE2.md).

One rank is exported as an ordered list of *parts*:

    qmd-n{k}.001.sql, qmd-n{k}.002.sql, ...

Part 001 starts by deleting every rank-k row, so importing a rank's parts in
order replaces that rank atomically-enough for our purposes (a rank is
idempotent as a whole, never part by part). Statements are cut by *bytes*
(D1 rejects statements over 100 KB) and parts by bytes too, so a D_10 orbit
of ~136k labeled matrices streams to disk without ever being one string.

Tables written (schema v2):
    mutation_classes, labelings, frontier_quivers, quivers, rank_stats
(`class_nicknames` is curated separately — scripts/nicknames.py — and is
never touched here.)

Resumability: manifest.json records, per rank, every part's sha256 and the
sha256 of each lower-rank acyclicity checkpoint the rank consumed. A rank is
skipped on re-run only if all of those still match; regenerating rank j
therefore invalidates every rank above it (the mutation-acyclic subquiver
fallback reads lower-rank results).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
from typing import Iterable, Iterator, Optional

from qmd import __version__ as PIPELINE_VERSION
from qmd import dynkin, invariants
from qmd import local_acyclicity as la
from qmd.class_properties import TRI, la_bounds, resolve_mutation_acyclic
from qmd.core import (
    GenerationResult,
    MutationClassResult,
    is_acyclic,
    is_connected,
    max_edge,
    quiver_id,
    run_generation,
    to_lists,
)

MANIFEST_NAME = "manifest.json"

# D1 limits (docs/PHASE2.md §0). Statements are cut well under the hard cap.
D1_STATEMENT_LIMIT = 100_000
STATEMENT_BYTES = 90_000
DEFAULT_PART_BYTES = 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def _finiteness(mc: MutationClassResult, n: int, bound: int) -> tuple:
    """(is_finite_confirmed, is_infinite_confirmed, is_infinite_expected, provenance-extra)."""
    if mc.exploration == "complete":
        return True, False, False, {}
    if mc.exploration == "bound" and n >= 3 and bound >= 2:
        return False, True, False, {
            "is_infinite_confirmed": {
                "method": (f"Derksen–Owen: a bounded mutation reached "
                           f"|b_ij| >= {bound + 1} (>= 3) at rank {n} >= 3"),
            }
        }
    # Truncated (or a bound crossing that proves nothing): everything unknown.
    return None, None, None, {
        "exploration": {
            "state": mc.exploration,
            "method": ("BFS stopped by the node cap; the class is only partially "
                       "explored and its finiteness is unknown"
                       if mc.exploration == "truncated" else
                       "a mutation crossed the weight bound at a rank where that "
                       "proves nothing"),
        }
    }


def build_rank_rows(result: GenerationResult, n: int,
                    known_acyclicity: Optional[dict] = None, *,
                    bound: int = 2, node_cap: Optional[int] = None) -> dict:
    """
    Compute the skinny rows for rank `n` and keep handles to the class objects
    so the heavy per-class rows (labelings, frontier) can be streamed later.

    Returns {
      "mutation_classes": [row, ...],   sorted by id
      "quivers":          [row, ...],   sorted by id (with labeling_offset)
      "rank_stats":       row,
      "classes":          {mc_id: MutationClassResult},   # for streaming
      "acyclicity_by_qid": {qid: True|False|None},        # checkpoint payload
    }
    """
    if bound < 2:
        raise ValueError("bound must be >= 2 (the seed set is {0, ±1, ±2} at least)")

    classes = {mc_id: mc for mc_id, mc in result.classes.items()
               if len(mc.canonical_rep) == n}
    quivers = {qid: m for qid, m in result.quivers.items() if len(m) == n}

    mut_acyclic = resolve_mutation_acyclic(
        [
            (mc_id, len(mc.canonical_rep),
             invariants.class_is_mutation_acyclic(mc.labeled_quivers, mc.is_open),
             mc.labeled_quivers, mc.quiver_ids)
            for mc_id, mc in classes.items()
        ],
        known=known_acyclicity,
    )

    class_rows: list[dict] = []
    for mc_id in sorted(classes):
        mc = classes[mc_id]
        rep = mc.canonical_rep
        is_open = mc.is_open
        complete = mc.exploration == "complete"
        dtype = dynkin.classify(rep) if complete else None

        bounds = la_bounds(is_open)
        b_state, b_w = la.banff_status(rep, **bounds)
        l_state, l_w = la.louise_status(rep, **bounds)
        p_state, p_w = la.p_prime_status(rep, **bounds)
        fin, inf, exp, extra = _finiteness(mc, n, bound)
        provenance = {
            "is_banff":   {"state": b_state, "witness": b_w},
            "is_louise":  {"state": l_state, "witness": l_w},
            "is_p_prime": {"state": p_state, "witness": p_w},
            **extra,
        }

        class_rows.append({
            "id": mc_id,
            "n": n,
            "canonical_matrix": to_lists(rep),
            "canonical_quiver_id": quiver_id(rep),
            "is_open": is_open,
            "exploration": mc.exploration,
            "class_size": mc.labeled_size,
            "distinct_quiver_count": mc.distinct_quiver_count,
            "merged_orbit_count": mc.merged_orbit_count,
            "dynkin_type": dtype,
            "label": dtype,
            "is_finite_confirmed": fin,
            "is_infinite_confirmed": inf,
            "is_infinite_expected": exp,
            "size_of_explored_frontier": len(mc.boundary_quivers),
            "is_mutation_acyclic": mut_acyclic[mc_id],
            "is_banff": TRI[b_state],
            "is_louise": TRI[l_state],
            "is_p_prime": TRI[p_state],
            "provenance": provenance,
        })

    # Per-quiver labeling counts (quiver_ids is parallel to labeled_quivers).
    labeling_counts: dict[str, int] = {}
    for mc in classes.values():
        for qid in mc.quiver_ids:
            labeling_counts[qid] = labeling_counts.get(qid, 0) + 1

    quiver_rows: list[dict] = []
    offset = 0
    for qid in sorted(quivers):
        matrix = quivers[qid]
        qi = invariants.quiver_invariants(matrix)
        count = labeling_counts.get(qid, 1)
        quiver_rows.append({
            "id": qid,
            "n": n,
            "exchange_matrix": to_lists(matrix),
            "mutation_class_id": result.membership.get(qid),
            "max_edge": max_edge(matrix),
            "is_acyclic": is_acyclic(matrix),
            "is_connected": is_connected(matrix),
            "is_bipartite": qi["is_bipartite"],
            "is_abundant": qi["is_abundant"],
            "is_planar": qi["is_planar"],
            "labeling_count": count,
            "labeling_offset": offset,
            "representation_type": qi["representation_type"],
            "symmetry_group": qi["symmetry_group"],
        })
        offset += count

    rank_stats = {
        "n": n,
        "quiver_count": len(quiver_rows),
        "labeled_quiver_count": sum(mc.labeled_size for mc in classes.values()),
        "class_count": len(class_rows),
        "bound": bound,
        "node_cap": node_cap,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_version": PIPELINE_VERSION,
    }

    acyclicity_by_qid = {
        qid: mut_acyclic[mc_id]
        for mc_id, mc in classes.items()
        for qid in set(mc.quiver_ids)
    }

    return {
        "mutation_classes": class_rows,
        "quivers": quiver_rows,
        "rank_stats": rank_stats,
        "classes": classes,
        "acyclicity_by_qid": acyclicity_by_qid,
    }


# ---------------------------------------------------------------------------
# SQL rendering (streaming, byte-bounded)
# ---------------------------------------------------------------------------

def _lit(v) -> str:
    """SQLite literal for a Python value; dicts/lists become compact JSON text."""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (dict, list, tuple)):
        v = json.dumps(v, separators=(",", ":"))
    return "'" + str(v).replace("'", "''") + "'"


_MC_COLUMNS = [
    "id", "n", "canonical_matrix", "canonical_quiver_id", "is_open", "exploration",
    "class_size", "distinct_quiver_count", "merged_orbit_count", "dynkin_type",
    "label", "is_finite_confirmed", "is_infinite_confirmed",
    "is_infinite_expected", "size_of_explored_frontier", "is_mutation_acyclic",
    "is_banff", "is_louise", "is_p_prime", "provenance",
]
_LABELING_COLUMNS = ["mutation_class_id", "ord", "qmd_id", "matrix"]
_FRONTIER_COLUMNS = ["mutation_class_id", "ord", "matrix"]
_QUIVER_COLUMNS = [
    "id", "n", "exchange_matrix", "mutation_class_id", "max_edge",
    "is_acyclic", "is_connected", "is_bipartite", "is_abundant", "is_planar",
    "labeling_count", "labeling_offset", "representation_type", "symmetry_group",
]
_STATS_COLUMNS = ["n", "quiver_count", "labeled_quiver_count", "class_count",
                  "bound", "node_cap", "generated_at", "pipeline_version"]


def _insert_stmts(table: str, columns: list[str], rows: Iterable,
                  stmt_bytes: int = STATEMENT_BYTES) -> Iterator[str]:
    """
    Multi-row INSERT statements, each under `stmt_bytes` (UTF-8). `rows` may
    be any iterable of dicts (or column-order tuples) — it is consumed lazily.
    """
    head = f"INSERT INTO {table} ({', '.join(columns)}) VALUES\n"
    head_len = len(head.encode("utf-8"))
    buf: list[str] = []
    size = head_len
    for r in rows:
        vals = r if isinstance(r, (list, tuple)) else [r[c] for c in columns]
        piece = "(" + ", ".join(_lit(v) for v in vals) + ")"
        plen = len(piece.encode("utf-8")) + 2
        if head_len + plen > D1_STATEMENT_LIMIT:
            raise ValueError(f"single {table} row exceeds the D1 statement limit "
                             f"({plen} bytes)")
        if buf and size + plen > stmt_bytes:
            yield head + ",\n".join(buf) + ";"
            buf, size = [], head_len
        buf.append(piece)
        size += plen
    if buf:
        yield head + ",\n".join(buf) + ";"


def _labeling_rows(classes: dict) -> Iterator[tuple]:
    for mc_id in sorted(classes):
        mc = classes[mc_id]
        for ord_, (m, qid) in enumerate(zip(mc.labeled_quivers, mc.quiver_ids)):
            yield (mc_id, ord_, qid, to_lists(m))


def _frontier_rows(classes: dict) -> Iterator[tuple]:
    for mc_id in sorted(classes):
        for ord_, m in enumerate(classes[mc_id].boundary_quivers):
            yield (mc_id, ord_, to_lists(m))


def iter_rank_statements(n: int, rows: dict, *, bound: int) -> Iterator[str]:
    """
    Every SQL statement for one rank, in import order: deletes, then parents
    before children (quivers and labelings reference mutation_classes).
    No BEGIN/COMMIT (D1 applies a file as a batch and rejects explicit
    transactions).
    """
    yield f"-- Quiver Mutation Database — rank {n} (bound |b_ij| <= {bound}, pipeline {PIPELINE_VERSION})."
    yield "-- Generated by: python scripts/populate.py --export-d1 <dir>"
    yield "-- Import with:  scripts/import-d1.sh <dir> [--remote]   (parts in order!)"
    yield (f"DELETE FROM labelings WHERE mutation_class_id IN "
           f"(SELECT id FROM mutation_classes WHERE n = {n});")
    yield (f"DELETE FROM frontier_quivers WHERE mutation_class_id IN "
           f"(SELECT id FROM mutation_classes WHERE n = {n});")
    yield f"DELETE FROM quivers WHERE n = {n};"
    yield f"DELETE FROM mutation_classes WHERE n = {n};"
    yield f"DELETE FROM rank_stats WHERE n = {n};"
    yield from _insert_stmts("mutation_classes", _MC_COLUMNS, rows["mutation_classes"])
    yield from _insert_stmts("quivers", _QUIVER_COLUMNS, rows["quivers"])
    yield from _insert_stmts("labelings", _LABELING_COLUMNS, _labeling_rows(rows["classes"]))
    yield from _insert_stmts("frontier_quivers", _FRONTIER_COLUMNS, _frontier_rows(rows["classes"]))
    yield from _insert_stmts("rank_stats", _STATS_COLUMNS, [rows["rank_stats"]])


def render_rank_sql(n: int, rows: dict, *, bound: int) -> str:
    """The whole rank as one string (tests / small ranks)."""
    return "\n".join(iter_rank_statements(n, rows, bound=bound)) + "\n"


class _PartWriter:
    """Writes statements into numbered part files, each under `part_bytes`."""

    def __init__(self, out_dir: str, base: str, part_bytes: int):
        self.out_dir, self.base, self.part_bytes = out_dir, base, part_bytes
        self.parts: list[dict] = []
        self._fh: Optional[io.TextIOWrapper] = None
        self._tmp = ""
        self._size = 0
        self._hash = hashlib.sha256()

    def _open(self) -> None:
        idx = len(self.parts) + 1
        name = f"{self.base}.{idx:03d}.sql"
        self._tmp = os.path.join(self.out_dir, name + ".tmp")
        self._fh = open(self._tmp, "w", encoding="utf-8")
        self._size, self._hash = 0, hashlib.sha256()
        self.parts.append({"file": name})

    def write(self, stmt: str) -> None:
        data = (stmt + "\n").encode("utf-8")
        if self._fh is not None and self._size + len(data) > self.part_bytes and self._size > 0:
            self._close_part()
        if self._fh is None:
            self._open()
        assert self._fh is not None
        self._fh.write(stmt + "\n")
        self._hash.update(data)
        self._size += len(data)

    def _close_part(self) -> None:
        assert self._fh is not None
        self._fh.close()
        final = os.path.join(self.out_dir, self.parts[-1]["file"])
        os.replace(self._tmp, final)
        self.parts[-1].update(sha256=self._hash.hexdigest(), bytes=self._size)
        self._fh = None

    def close(self) -> list[dict]:
        if self._fh is not None:
            self._close_part()
        return self.parts


def write_rank_sql(out_dir: str, n: int, rows: dict, *, bound: int,
                   part_bytes: int = DEFAULT_PART_BYTES) -> list[dict]:
    """Stream the rank to `qmd-n{n}.NNN.sql` parts; returns the manifest entries."""
    w = _PartWriter(out_dir, f"qmd-n{n}", part_bytes)
    for stmt in iter_rank_statements(n, rows, bound=bound):
        w.write(stmt)
    return w.close()


# ---------------------------------------------------------------------------
# Checkpointed per-rank driver
# ---------------------------------------------------------------------------

def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _up_to_date(out_dir: str, entry: Optional[dict], depends_on: dict) -> bool:
    if not entry or "parts" not in entry:
        return False
    for part in entry["parts"]:
        if _sha256_file(os.path.join(out_dir, part["file"])) != part.get("sha256"):
            return False
    return entry.get("depends_on", {}) == depends_on


def _remove_stale_parts(out_dir: str, n: int) -> None:
    prefix = f"qmd-n{n}."
    for name in os.listdir(out_dir):
        if name.startswith(prefix) and name.endswith(".sql"):
            os.remove(os.path.join(out_dir, name))


def export_ranks(out_dir: str, *, max_vertices: int, bound: int,
                 ranks: Optional[Iterable[int]] = None, force: bool = False,
                 node_cap: Optional[int] = None,
                 part_bytes: int = DEFAULT_PART_BYTES, log=print) -> None:
    """
    Generate and export ranks 1..max_vertices (or the given `ranks`) as
    multipart SQL, checkpointed so an interrupted run resumes where it left off.

    Ranks are processed in ascending order because the is_mutation_acyclic
    subquiver fallback consumes lower-rank results; each rank's quiver-level
    states are persisted to acyclicity-n{k}.json and their hashes recorded, so
    regenerating a lower rank invalidates every rank above it.
    """
    if bound < 2:
        raise SystemExit("--bound must be >= 2")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    manifest = _load_json(manifest_path) or {"bound": bound, "ranks": {}}

    settings = {"bound": bound, "node_cap": node_cap}
    if (manifest.get("bound"), manifest.get("node_cap")) != (bound, node_cap):
        if manifest.get("ranks") and not force:
            raise SystemExit(
                f"{manifest_path} was generated with bound={manifest.get('bound')}, "
                f"node_cap={manifest.get('node_cap')}; requested {settings}. "
                "Re-run with --force to regenerate everything."
            )
        manifest = {**settings, "ranks": {}}
    manifest.update(settings)
    manifest["pipeline_version"] = PIPELINE_VERSION

    todo = sorted(ranks) if ranks else list(range(1, max_vertices + 1))

    for n in todo:
        entry = manifest["ranks"].get(str(n))

        # Lower-rank checkpoints this rank depends on (and their hashes).
        known: dict = {}
        depends_on: dict = {}
        for j in range(1, n):
            ck_path = os.path.join(out_dir, f"acyclicity-n{j}.json")
            ck = _load_json(ck_path)
            if ck is None:
                raise SystemExit(
                    f"rank {n} needs acyclicity-n{j}.json (lower-rank "
                    f"checkpoint) but it is missing — export rank {j} first "
                    "(or re-run without --ranks)."
                )
            known.update(ck)
            depends_on[f"acyclicity-n{j}.json"] = _sha256_file(ck_path)

        if not force and _up_to_date(out_dir, entry, depends_on):
            log(f"  rank {n}: up to date ({len(entry['parts'])} part(s)), skipping")
            continue
        if entry and not force:
            log(f"  rank {n}: export or its checkpoints changed, regenerating")

        log(f"  rank {n}: generating (bound |b_ij| <= {bound}"
            f"{f', node cap {node_cap}' if node_cap else ''}) ...")
        result = run_generation(max_vertices=n, bound=bound, ranks=[n], node_cap=node_cap)
        rows = build_rank_rows(result, n, known_acyclicity=known, bound=bound, node_cap=node_cap)

        _remove_stale_parts(out_dir, n)
        parts = write_rank_sql(out_dir, n, rows, bound=bound, part_bytes=part_bytes)
        _atomic_write(os.path.join(out_dir, f"acyclicity-n{n}.json"),
                      json.dumps(rows["acyclicity_by_qid"], sort_keys=True))
        stats = rows["rank_stats"]
        manifest["ranks"][str(n)] = {
            "parts": parts,
            "depends_on": depends_on,
            "quiver_count": stats["quiver_count"],
            "labeled_quiver_count": stats["labeled_quiver_count"],
            "class_count": stats["class_count"],
            "truncated_classes": sum(
                1 for r in rows["mutation_classes"] if r["exploration"] == "truncated"),
            "generated_at": stats["generated_at"],
        }
        _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
        log(f"  rank {n}: wrote {len(parts)} part(s), "
            f"{sum(p['bytes'] for p in parts) / 1e6:.1f} MB "
            f"({stats['quiver_count']} quivers, {stats['class_count']} classes, "
            f"{stats['labeled_quiver_count']} labelings)")
