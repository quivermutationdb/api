"""
qmd/d1_export.py

GenerationResult (+ quiver-only rows) -> per-rank SQL for Cloudflare D1
(schema v3: docs/PHASE2.md, docs/PHASE3.md).

One rank is exported as an ordered list of *parts*, one sequence per shard:

    qmd-n{k}.main.001.sql, ...            ranks that live in the main database
    qmd-n{k}.s0.001.sql, qmd-n{k}.s1.001.sql, ...   split ranks (data/shards.json)
    qmd-n{k}.stats.sql                    the rank_stats row (main database only)

Part 001 of every shard sequence starts by deleting that rank's rows in that
shard, so a rank is idempotent as a whole when its parts are imported in
order (scripts/import-d1.sh reads the manifest and targets the right
database). Statements are cut by *bytes* (D1 rejects statements over 100 KB)
and parts by bytes, so nothing is ever one big string.

Rows per table (schema v3):
    quivers          one row per unlabeled quiver: compact matrix, invariants,
                     mutation_finite (three-state), mutation_class_id or NULL
    mutation_classes one row per *explored* class
    labelings        every labeled matrix — only for completely explored
                     (mutation-finite) classes
    rank_stats       aggregates + provenance (main database)
(`class_nicknames` is curated separately — scripts/nicknames.py.)

The census is of CONNECTED quivers only (a disconnected quiver is a disjoint
union; its class is the product of its components' classes).

Finiteness of a quiver's class comes from three sources, all sound:
  * the quiver itself has an entry |b_ij| >= 3 at rank >= 3  -> infinite
    (Derksen–Owen), no exploration needed;
  * its explored class crossed |b_ij| <= 2                    -> infinite;
  * its explored class was drained                           -> finite;
  otherwise NULL (unknown). The exploration bound is always 2 — the wall at 3
  IS the Derksen–Owen witness — whatever the cell's height h.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
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
)
from qmd.encoding import encode_upper

MANIFEST_NAME = "manifest.json"
EXPLORE_BOUND = 2          # never a knob: the wall at |b_ij| = 3 is the Derksen–Owen witness

# D1 limits (docs/PHASE2.md §0). Statements are cut well under the hard cap.
D1_STATEMENT_LIMIT = 100_000
STATEMENT_BYTES = 90_000
DEFAULT_PART_BYTES = 64 * 1024 * 1024

SHARDS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "shards.json")


def _shards_config() -> dict:
    with open(SHARDS_FILE, encoding="utf-8") as f:
        return json.load(f)


def shard_of(id_: str, n: int, cfg: Optional[dict] = None) -> tuple[str, str]:
    """(shard key, database name) holding this id: ('main', 'qmd') or ('n6.0', 'qmd-n6-0')."""
    cfg = cfg or _shards_config()
    split = cfg.get("split", {}).get(str(n))
    if not split:
        return "main", cfg["main"]["database"]
    b = int(id_[id_.rfind(".") + 1], 16) % split["buckets"]
    return f"n{n}.{b}", split["databases"][b]["database"]


def shard_keys_for(n: int, cfg: Optional[dict] = None) -> list[tuple[str, str]]:
    cfg = cfg or _shards_config()
    split = cfg.get("split", {}).get(str(n))
    if not split:
        return [("main", cfg["main"]["database"])]
    return [(f"n{n}.{i}", d["database"]) for i, d in enumerate(split["databases"])]


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def derksen_owen_infinite(matrix, n: int) -> bool:
    """A rank >= 3 quiver with an entry |b_ij| >= 3 is mutation-infinite."""
    return n >= 3 and max_edge(matrix) >= 3


def _finiteness(mc: MutationClassResult, n: int) -> tuple:
    """(is_finite_confirmed, is_infinite_confirmed, is_infinite_expected, provenance-extra)."""
    if mc.exploration == "complete":
        return True, False, False, {}
    if mc.exploration == "bound" and n >= 3:
        return False, True, False, {
            "is_infinite_confirmed": {
                "method": (f"Derksen–Owen: a bounded mutation reached "
                           f"|b_ij| >= {EXPLORE_BOUND + 1} at rank {n} >= 3"),
            }
        }
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


def _class_job(args: tuple) -> tuple:
    """Worker: the per-class searches that only need the canonical rep."""
    mc_id, rep, is_open, complete, la_timeout = args
    dtype = dynkin.classify(rep, mc_id) if complete else None
    bounds = la_bounds(is_open)
    if la_timeout is not None and is_open:
        bounds["timeout"] = la_timeout
    if la_timeout == 0 and is_open:
        return (mc_id, dtype, ("unknown", None), ("unknown", None), ("unknown", None))
    return (mc_id, dtype,
            la.banff_status(rep, **bounds),
            la.louise_status(rep, **bounds),
            la.p_prime_status(rep, **bounds))


def _quiver_job(args: tuple) -> tuple:
    qid, matrix = args
    return qid, invariants.quiver_invariants(matrix)


def _pmap(fn, jobs: list, workers: int, progress=None, label: str = "", chunksize: int = 16):
    """Ordered results from a process pool (or inline when workers <= 1)."""
    if workers <= 1 or len(jobs) < 2:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    out = []
    with mp.get_context("fork").Pool(workers) as pool:
        for i, r in enumerate(pool.imap(fn, jobs, chunksize=chunksize), 1):
            out.append(r)
            if progress and (i % 2000 == 0 or i == len(jobs)):
                progress(label, i, len(jobs))
    return out


def _shard_counts(n: int, quiver_rows: list, class_rows: list) -> dict:
    cfg = _shards_config()
    out = {key: {"quivers": 0, "classes": 0} for key, _db in shard_keys_for(n, cfg)}
    for r in quiver_rows:
        out[shard_of(r["id"], n, cfg)[0]]["quivers"] += 1
    for r in class_rows:
        out[shard_of(r["id"], n, cfg)[0]]["classes"] += 1
    return out


def build_rank_rows(result: GenerationResult, n: int,
                    known_acyclicity: Optional[dict] = None, *,
                    bound: int = 2, node_cap: Optional[int] = None,
                    generator: str = "orderly", census_size: Optional[int] = None,
                    sample_size: Optional[int] = None,
                    extra_quivers: Optional[dict] = None,
                    la_timeout: Optional[float] = None,
                    workers: int = 1, progress=None) -> dict:
    """
    Compute the rows for rank `n`.

    `result` holds the explored classes and their member quivers.
    `extra_quivers` = {qid: (matrix, mutation_finite)} adds quiver-only rows
    (no class): Derksen–Owen-infinite seeds, label-only explorations, etc.

    Returns {
      "mutation_classes": [row, ...],   sorted by id
      "quivers":          [row, ...],   sorted by id
      "rank_stats":       row,
      "classes":          {mc_id: MutationClassResult},   # for labelings streaming
      "acyclicity_by_qid": {qid: True|False|None},        # checkpoint payload
    }
    """
    classes = {mc_id: mc for mc_id, mc in result.classes.items()
               if len(mc.canonical_rep) == n}
    quivers = {qid: m for qid, m in result.quivers.items() if len(m) == n}
    extra = dict(extra_quivers or {})
    for qid in list(extra):
        if qid in quivers:
            del extra[qid]

    mut_acyclic = resolve_mutation_acyclic(
        [
            (mc_id, len(mc.canonical_rep),
             invariants.class_is_mutation_acyclic(mc.members, mc.is_open),
             mc.members, mc.quiver_ids)
            for mc_id, mc in classes.items()
        ],
        known=known_acyclicity,
    )

    # Warm the Dynkin reference in the parent so forked workers inherit it.
    if any(mc.exploration == "complete" for mc in classes.values()):
        dynkin.reference_for(n)

    ordered = sorted(classes)
    searched = _pmap(_class_job, [
        (mc_id, classes[mc_id].canonical_rep, classes[mc_id].is_open,
         classes[mc_id].exploration == "complete", la_timeout) for mc_id in ordered
    ], workers, progress, "classes")

    class_rows: list[dict] = []
    finite_by_class: dict[str, Optional[bool]] = {}
    for mc_id, dtype, (b_state, b_w), (l_state, l_w), (p_state, p_w) in searched:
        mc = classes[mc_id]
        rep = mc.canonical_rep
        fin, inf, exp, extra_prov = _finiteness(mc, n)
        finite_by_class[mc_id] = True if fin else (False if inf else None)
        provenance = {
            "is_banff":   {"state": b_state, "witness": b_w},
            "is_louise":  {"state": l_state, "witness": l_w},
            "is_p_prime": {"state": p_state, "witness": p_w},
            **extra_prov,
        }
        class_rows.append({
            "id": mc_id,
            "n": n,
            "canonical_matrix": encode_upper(rep),
            "canonical_quiver_id": quiver_id(rep),
            "is_open": mc.is_open,
            "exploration": mc.exploration,
            "class_size": mc.labeled_size,          # None: labeled orbit not stored
            "distinct_quiver_count": mc.distinct_quiver_count,
            "merged_orbit_count": mc.merged_orbit_count,
            "dynkin_type": dtype,
            "label": dtype,
            "is_finite_confirmed": fin,
            "is_infinite_confirmed": inf,
            "is_infinite_expected": exp,
            "size_of_explored_frontier": mc.boundary_count,
            "is_mutation_acyclic": mut_acyclic[mc_id],
            "is_banff": TRI[b_state],
            "is_louise": TRI[l_state],
            "is_p_prime": TRI[p_state],
            "provenance": provenance,
        })

    # Per-quiver labeling counts, known only where the labeled orbit was computed.
    labeling_counts: dict[str, int] = {}
    for mc in classes.values():
        for qid in mc.labeled_ids:
            labeling_counts[qid] = labeling_counts.get(qid, 0) + 1

    all_matrices = {**{qid: m for qid, m in quivers.items()},
                    **{qid: m for qid, (m, _f) in extra.items()}}
    qinv = dict(_pmap(_quiver_job, [(qid, all_matrices[qid]) for qid in sorted(all_matrices)],
                      workers, progress, "quivers", chunksize=256))

    quiver_rows: list[dict] = []
    for qid in sorted(all_matrices):
        matrix = all_matrices[qid]
        qi = qinv[qid]
        if not is_connected(matrix):
            # Cannot happen (seeds are connected and mutation preserves
            # connectedness) — but the census guarantee is worth a hard stop.
            raise RuntimeError(f"disconnected quiver {qid} reached the exporter")
        if qid in quivers:
            mc_id = result.membership.get(qid)
            finite = finite_by_class.get(mc_id) if mc_id else None
            count = labeling_counts.get(qid) if mc_id and classes[mc_id].has_labelings else None
        else:
            mc_id = None
            finite = extra[qid][1]
            count = None
        if finite is None and derksen_owen_infinite(matrix, n):
            finite = False
        quiver_rows.append({
            "id": qid,
            "n": n,
            "exchange_matrix": encode_upper(matrix),
            "mutation_class_id": mc_id,
            "mutation_finite": finite,
            "max_edge": max_edge(matrix),
            "is_acyclic": is_acyclic(matrix),
            "is_connected": is_connected(matrix),
            "is_bipartite": qi["is_bipartite"],
            "is_abundant": qi["is_abundant"],
            "is_planar": qi["is_planar"],
            "labeling_count": count,
            "representation_type": qi["representation_type"],
            "symmetry_group": qi["symmetry_group"],
        })

    rank_stats = {
        "n": n,
        "quiver_count": len(quiver_rows),
        "labeled_quiver_count": sum(mc.labeled_size or 0 for mc in classes.values()),
        "class_count": len(class_rows),
        "bound": bound,
        "node_cap": node_cap,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_version": PIPELINE_VERSION,
        "generator": generator if sample_size is None else f"sample:{sample_size}",
        "census_size": census_size,
        "shard_counts": _shard_counts(n, quiver_rows, class_rows),
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
# SQL rendering (streaming, byte-bounded, per shard)
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
_QUIVER_COLUMNS = [
    "id", "n", "exchange_matrix", "mutation_class_id", "mutation_finite", "max_edge",
    "is_acyclic", "is_connected", "is_bipartite", "is_abundant", "is_planar",
    "labeling_count", "representation_type", "symmetry_group",
]
_STATS_COLUMNS = ["n", "quiver_count", "labeled_quiver_count", "class_count",
                  "bound", "node_cap", "generated_at", "pipeline_version",
                  "generator", "census_size", "shard_counts"]


def _insert_stmts(table: str, columns: list[str], rows: Iterable,
                  stmt_bytes: int = STATEMENT_BYTES) -> Iterator[str]:
    """Multi-row INSERTs, each under `stmt_bytes` (UTF-8); `rows` is consumed lazily."""
    head = f"INSERT INTO {table} ({', '.join(columns)}) VALUES\n"
    head_len = len(head.encode("utf-8"))
    buf: list[str] = []
    size = head_len
    for r in rows:
        vals = r if isinstance(r, (list, tuple)) else [r[c] for c in columns]
        piece = "(" + ", ".join(_lit(v) for v in vals) + ")"
        plen = len(piece.encode("utf-8")) + 2
        if head_len + plen > D1_STATEMENT_LIMIT:
            raise ValueError(f"single {table} row exceeds the D1 statement limit ({plen} bytes)")
        if buf and size + plen > stmt_bytes:
            yield head + ",\n".join(buf) + ";"
            buf, size = [], head_len
        buf.append(piece)
        size += plen
    if buf:
        yield head + ",\n".join(buf) + ";"


def _labeling_rows(classes: dict, keys: set) -> Iterator[tuple]:
    """Labelings of the classes in `keys` whose labeled orbit was computed."""
    for mc_id in sorted(keys):
        mc = classes[mc_id]
        for ord_, (m, qid) in enumerate(zip(mc.labeled_quivers, mc.labeled_ids)):
            yield (mc_id, ord_, qid, encode_upper(m))


def _header(n: int, bound: int, shard_key: str) -> list[str]:
    return [
        f"-- Quiver Mutation Database — rank {n}, shard {shard_key} "
        f"(cell |b_ij| <= {bound}, explore bound {EXPLORE_BOUND}, pipeline {PIPELINE_VERSION}).",
        "-- Generated by: python scripts/populate.py --export-d1 <dir>",
        "-- Import with:  scripts/import-d1.sh <dir> [--remote]   (parts in order!)",
    ]


def iter_shard_statements(n: int, rows: dict, shard_key: str, *, bound: int) -> Iterator[str]:
    """All statements for one shard of one rank: deletes, classes, quivers, labelings."""
    cfg = _shards_config()
    mc_ids = {r["id"] for r in rows["mutation_classes"] if shard_of(r["id"], n, cfg)[0] == shard_key}
    yield from _header(n, bound, shard_key)
    yield (f"DELETE FROM labelings WHERE mutation_class_id IN "
           f"(SELECT id FROM mutation_classes WHERE n = {n});")
    yield f"DELETE FROM quivers WHERE n = {n};"
    yield f"DELETE FROM mutation_classes WHERE n = {n};"
    yield from _insert_stmts("mutation_classes", _MC_COLUMNS,
                             (r for r in rows["mutation_classes"] if r["id"] in mc_ids))
    yield from _insert_stmts("quivers", _QUIVER_COLUMNS,
                             (r for r in rows["quivers"] if shard_of(r["id"], n, cfg)[0] == shard_key))
    yield from _insert_stmts("labelings", _LABELING_COLUMNS, _labeling_rows(rows["classes"], mc_ids))


def iter_stats_statements(n: int, rows: dict, *, bound: int) -> Iterator[str]:
    yield from _header(n, bound, "main (rank_stats)")
    yield f"DELETE FROM rank_stats WHERE n = {n};"
    yield from _insert_stmts("rank_stats", _STATS_COLUMNS, [rows["rank_stats"]])


def render_rank_sql(n: int, rows: dict, *, bound: int) -> dict[str, str]:
    """{shard key or 'stats': SQL text} for the whole rank (tests / small ranks)."""
    out = {key: "\n".join(iter_shard_statements(n, rows, key, bound=bound)) + "\n"
           for key, _db in shard_keys_for(n)}
    out["stats"] = "\n".join(iter_stats_statements(n, rows, bound=bound)) + "\n"
    return out


class _PartWriter:
    """Writes statements into numbered part files, each under `part_bytes`."""

    def __init__(self, out_dir: str, base: str, part_bytes: int):
        self.out_dir, self.base, self.part_bytes = out_dir, base, part_bytes
        self.parts: list[dict] = []
        self._fh = None
        self._tmp = ""
        self._size = 0
        self._hash = hashlib.sha256()

    def _open(self) -> None:
        name = f"{self.base}.{len(self.parts) + 1:03d}.sql"
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
        self._fh.write(stmt + "\n")
        self._hash.update(data)
        self._size += len(data)

    def _close_part(self) -> None:
        self._fh.close()
        os.replace(self._tmp, os.path.join(self.out_dir, self.parts[-1]["file"]))
        self.parts[-1].update(sha256=self._hash.hexdigest(), bytes=self._size)
        self._fh = None

    def close(self) -> list[dict]:
        if self._fh is not None:
            self._close_part()
        return self.parts


def write_rank_sql(out_dir: str, n: int, rows: dict, *, bound: int,
                   part_bytes: int = DEFAULT_PART_BYTES) -> list[dict]:
    """Stream the rank into per-shard part files; returns the manifest entries."""
    parts: list[dict] = []
    for key, database in shard_keys_for(n):
        suffix = "main" if key == "main" else "s" + key.split(".")[1]
        w = _PartWriter(out_dir, f"qmd-n{n}.{suffix}", part_bytes)
        for stmt in iter_shard_statements(n, rows, key, bound=bound):
            w.write(stmt)
        parts.extend({**p, "shard": key, "database": database} for p in w.close())
    cfg = _shards_config()
    w = _PartWriter(out_dir, f"qmd-n{n}.stats", part_bytes)
    for stmt in iter_stats_statements(n, rows, bound=bound):
        w.write(stmt)
    parts.extend({**p, "shard": "main", "database": cfg["main"]["database"]} for p in w.close())
    return parts


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


def _seeds_for(n: int, bound: int, generator: str, sample: Optional[int],
               sample_seed: int, workers: int, log) -> tuple[list, int]:
    """Connected phase-1 seeds for a rank plus the exact size of the connected cell (n, bound)."""
    from qmd import census
    size = census.count_connected_quivers(n, bound)
    from qmd.core import is_connected
    if generator == "brute":
        from qmd.core import generate_seed_quivers
        return [m for m in generate_seed_quivers(n, bound, ranks=[n]) if is_connected(m)], size
    if generator == "orderly":
        return census.census_seeds(
            n, bound, workers=workers, connected_only=True,
            progress=lambda k, c: log(f"    orderly level {k}: {c} quivers (all, before the connected filter)")), size
    if generator == "sample":
        if not sample:
            raise SystemExit("--sample N is required with --generator sample")
        if sample >= size:
            log(f"    sample {sample} >= cell size {size}; enumerating instead")
            return census.census_seeds(n, bound, workers=workers, connected_only=True), size
        return census.sample_cell(n, bound, sample, seed=sample_seed, connected_only=True), size
    raise SystemExit(f"unknown generator {generator!r}")


def _curated_seeds(n: int, log) -> list:
    """Extra seeds from data/seeds.json (e.g. E8), canonicalised; skipped if absent."""
    from qmd.core import canonical_form, to_matrix
    path = os.path.join(os.path.dirname(__file__), "..", "data", "seeds.json")
    doc = _load_json(path) or {}
    from qmd.core import is_connected
    out = []
    for e in doc.get("seeds", []):
        m = to_matrix(e["matrix"])
        if len(m) != n:
            continue
        if not is_connected(m):
            raise SystemExit(f"data/seeds.json: {e.get('name', '?')} is disconnected; the census is connected-only")
        out.append(canonical_form(m))
    if out:
        log(f"    {len(out)} curated seed(s) from data/seeds.json")
    return out


def export_ranks(out_dir: str, *, max_vertices: int, bound: int,
                 ranks: Optional[Iterable[int]] = None, force: bool = False,
                 node_cap: Optional[int] = None,
                 part_bytes: int = DEFAULT_PART_BYTES,
                 generator: str = "orderly", sample: Optional[int] = None,
                 sample_seed: int = 0, workers: int = 1,
                 la_timeout: Optional[float] = None, log=print) -> None:
    """
    Generate and export ranks 1..max_vertices (or `ranks`) as multipart SQL,
    checkpointed so an interrupted run resumes where it left off.

    Seeds with an entry |b_ij| >= 3 (rank >= 3) are Derksen–Owen-infinite and
    are stored as quiver-only rows without exploration; all other seeds are
    explored with bound EXPLORE_BOUND (= 2) and the node cap.
    """
    if bound < 1:
        raise SystemExit("--bound must be >= 1")
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    manifest = _load_json(manifest_path) or {"ranks": {}}
    manifest.setdefault("ranks", {})
    manifest["pipeline_version"] = PIPELINE_VERSION

    todo = sorted(ranks) if ranks else list(range(1, max_vertices + 1))

    for n in todo:
        settings = {"bound": bound, "node_cap": node_cap, "generator": generator,
                    "sample": sample if generator == "sample" else None,
                    "la_timeout": la_timeout, "schema": 3}
        entry = manifest["ranks"].get(str(n))

        known: dict = {}
        depends_on: dict = {}
        for j in range(1, n):
            ck_path = os.path.join(out_dir, f"acyclicity-n{j}.json")
            ck = _load_json(ck_path)
            if ck is None:
                raise SystemExit(
                    f"rank {n} needs acyclicity-n{j}.json (lower-rank checkpoint) "
                    f"but it is missing — export rank {j} first (or re-run without --ranks).")
            known.update(ck)
            depends_on[f"acyclicity-n{j}.json"] = _sha256_file(ck_path)

        if not force and entry and entry.get("settings") == settings and _up_to_date(out_dir, entry, depends_on):
            log(f"  rank {n}: up to date ({len(entry['parts'])} part(s)), skipping")
            continue
        if entry and not force:
            log(f"  rank {n}: settings, export or checkpoints changed, regenerating")

        log(f"  rank {n}: generating (cell |b_ij| <= {bound}, explore bound {EXPLORE_BOUND}"
            f"{f', node cap {node_cap}' if node_cap else ''}, {generator}"
            f"{f', {workers} workers' if workers > 1 else ''}) ...")
        seeds, census_size = _seeds_for(n, bound, generator, sample, sample_seed, workers, log)
        curated = _curated_seeds(n, log)
        seeds = sorted(set(seeds) | set(curated))
        sample_size = len(seeds) if generator == "sample" else None
        log(f"    {len(seeds)} seed quivers (cell size {census_size:,})")

        # Derksen–Owen shortcut: no BFS for seeds that already contain a 3.
        explore = [s for s in seeds if not derksen_owen_infinite(s, n)]
        skipped = {quiver_id(s): (s, False) for s in seeds if derksen_owen_infinite(s, n)}
        if skipped:
            log(f"    {len(skipped)} seeds have |b_ij| >= 3: mutation-infinite by Derksen–Owen, not explored")

        def prog(stage, i, total):
            if i == total or i % max(1, total // 10) == 0:
                log(f"    {stage}: {i}/{total}")
        # Rank <= 2 never changes weights under mutation, so a rank-2 seed with a
        # large entry is explored at its own weight (Derksen–Owen needs rank >= 3).
        explore_bound = EXPLORE_BOUND if n >= 3 else max(EXPLORE_BOUND, bound)
        result = run_generation(max_vertices=n, bound=explore_bound, ranks=[n], node_cap=node_cap,
                                seeds=explore, workers=workers, progress=prog)
        log(f"    {len(result.quivers)} explored quivers in {len(result.classes)} classes; computing invariants ...")
        rows = build_rank_rows(result, n, known_acyclicity=known, bound=bound, node_cap=node_cap,
                               generator=generator, census_size=census_size, sample_size=sample_size,
                               extra_quivers=skipped, la_timeout=la_timeout,
                               workers=workers, progress=prog)

        _remove_stale_parts(out_dir, n)
        parts = write_rank_sql(out_dir, n, rows, bound=bound, part_bytes=part_bytes)
        _atomic_write(os.path.join(out_dir, f"acyclicity-n{n}.json"),
                      json.dumps(rows["acyclicity_by_qid"], sort_keys=True))
        stats = rows["rank_stats"]
        manifest["ranks"][str(n)] = {
            "parts": parts,
            "depends_on": depends_on,
            "settings": settings,
            "quiver_count": stats["quiver_count"],
            "labeled_quiver_count": stats["labeled_quiver_count"],
            "class_count": stats["class_count"],
            "truncated_classes": sum(1 for r in rows["mutation_classes"] if r["exploration"] == "truncated"),
            "census_size": census_size,
            "generated_at": stats["generated_at"],
        }
        _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
        log(f"  rank {n}: wrote {len(parts)} part(s), "
            f"{sum(p['bytes'] for p in parts) / 1e6:.1f} MB "
            f"({stats['quiver_count']} quivers, {stats['class_count']} classes, "
            f"{stats['labeled_quiver_count']} labelings)")
