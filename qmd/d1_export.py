"""
qmd/d1_export.py

Export mode for the Cloudflare migration: turn a GenerationResult into
self-contained SQL files — one per rank `n` — ready for

    wrangler d1 execute qmd --remote --file=dist/d1/qmd-n{k}.sql

(current wrangler has no `d1 import`; `d1 execute --file` is the import path).

Design points (see CLAUDE.md):

- One file per rank.  This pre-aligns the artifacts with the future per-`n`
  shards: each file only ever touches rank-n rows, so it can later be pointed
  at a rank-n shard unchanged.
- Self-contained.  Each file deletes and re-inserts its rank's rows (including
  the rank_stats aggregate), so re-importing a file is idempotent and importing
  the files in any order yields the same database.
- Resumable / checkpointable.  export_ranks() processes ranks in ascending
  order, writes each file atomically, and records per-rank state in
  manifest.json.  The subquiver fallback for is_mutation_acyclic needs the
  quiver-level states of lower ranks, so those are checkpointed per rank in
  acyclicity-n{k}.json and re-loaded on resume.
- No database driver required.  This module (and everything it imports) is
  pure Python — it runs on a bare cloud compute box with just the repo.

The row shapes mirror qmd/crud.upsert_generation_result exactly, targeting the
Drizzle schema in src/db/schema.ts (see drizzle/0000_init_schema.sql).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable, Optional

from qmd import dynkin, invariants
from qmd import local_acyclicity as la
from qmd.class_properties import TRI, la_bounds, resolve_mutation_acyclic
from qmd.core import (
    GenerationResult,
    is_acyclic,
    is_connected,
    max_edge,
    quiver_id,
    run_generation,
    to_lists,
)

MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# Row building (mirror of crud.upsert_generation_result, minus the ORM)
# ---------------------------------------------------------------------------

def build_rank_rows(result: GenerationResult, n: int,
                    known_acyclicity: Optional[dict] = None) -> dict:
    """
    Compute all D1 rows for rank `n` from a GenerationResult that contains
    (at least) that rank.

    Returns {
      "mutation_classes":        [row, ...],   sorted by id
      "mutation_class_payloads": [row, ...],   sorted by mutation_class_id
      "quivers":                 [row, ...],   sorted by id
      "rank_stats":              row,
      "acyclicity_by_qid":       {qid: True|False|None},   # checkpoint payload
    }
    """
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
    payload_rows: list[dict] = []
    for mc_id in sorted(classes):
        mc = classes[mc_id]
        rep = mc.canonical_rep
        is_open = mc.is_open
        dtype = None if is_open else dynkin.classify(rep)

        bounds = la_bounds(is_open)
        b_state, b_w = la.banff_status(rep, **bounds)
        l_state, l_w = la.louise_status(rep, **bounds)
        p_state, p_w = la.p_prime_status(rep, **bounds)
        provenance = {
            "is_banff":   {"state": b_state, "witness": b_w},
            "is_louise":  {"state": l_state, "witness": l_w},
            "is_p_prime": {"state": p_state, "witness": p_w},
        }
        if is_open:
            provenance["is_infinite_confirmed"] = {
                "method": "Derksen-Owen: a bounded mutation reached |b_ij| >= 3"
            }

        class_rows.append({
            "id": mc_id,
            "n": n,
            "canonical_matrix": to_lists(rep),
            "canonical_quiver_id": quiver_id(rep),
            "is_open": is_open,
            "class_size": mc.labeled_size,
            "distinct_quiver_count": mc.distinct_quiver_count,
            "merged_orbit_count": mc.merged_orbit_count,
            "dynkin_type": dtype,
            "label": dtype,
            "is_finite_confirmed": not is_open,
            "is_infinite_confirmed": is_open,
            "is_infinite_expected": False,
            "size_of_explored_frontier": len(mc.boundary_quivers),
            "is_mutation_acyclic": mut_acyclic[mc_id],
            "is_banff": TRI[b_state],
            "is_louise": TRI[l_state],
            "is_p_prime": TRI[p_state],
            "provenance": provenance,
        })
        payload_rows.append({
            "mutation_class_id": mc_id,
            "labeled_quivers": [
                {"qmd_id": qid, "matrix": to_lists(m)}
                for m, qid in zip(mc.labeled_quivers, mc.quiver_ids)
            ],
            "boundary_quivers": [to_lists(m) for m in mc.boundary_quivers],
        })

    # Per-quiver labeling counts (quiver_ids is parallel to labeled_quivers).
    labeling_counts: dict[str, int] = {}
    for mc in classes.values():
        for qid in mc.quiver_ids:
            labeling_counts[qid] = labeling_counts.get(qid, 0) + 1

    quiver_rows: list[dict] = []
    for qid in sorted(quivers):
        matrix = quivers[qid]
        qi = invariants.quiver_invariants(matrix)
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
            "labeling_count": labeling_counts.get(qid, 1),
            "representation_type": qi["representation_type"],
            "symmetry_group": qi["symmetry_group"],
        })

    rank_stats = {
        "n": n,
        "quiver_count": len(quiver_rows),
        "labeled_quiver_count": sum(mc.labeled_size for mc in classes.values()),
        "class_count": len(class_rows),
    }

    acyclicity_by_qid = {
        qid: mut_acyclic[mc_id]
        for mc_id, mc in classes.items()
        for qid in set(mc.quiver_ids)
    }

    return {
        "mutation_classes": class_rows,
        "mutation_class_payloads": payload_rows,
        "quivers": quiver_rows,
        "rank_stats": rank_stats,
        "acyclicity_by_qid": acyclicity_by_qid,
    }


# ---------------------------------------------------------------------------
# SQL rendering
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


def _insert_stmts(table: str, columns: list[str], rows: list[dict],
                  chunk: int) -> Iterable[str]:
    """INSERT statements, `chunk` rows each (D1 rejects huge single statements)."""
    for start in range(0, len(rows), chunk):
        values = ",\n".join(
            "(" + ", ".join(_lit(r[c]) for c in columns) + ")"
            for r in rows[start:start + chunk]
        )
        yield f"INSERT INTO {table} ({', '.join(columns)}) VALUES\n{values};"


_MC_COLUMNS = [
    "id", "n", "canonical_matrix", "canonical_quiver_id", "is_open",
    "class_size", "distinct_quiver_count", "merged_orbit_count", "dynkin_type",
    "label", "is_finite_confirmed", "is_infinite_confirmed",
    "is_infinite_expected", "size_of_explored_frontier", "is_mutation_acyclic",
    "is_banff", "is_louise", "is_p_prime", "provenance",
]
_PAYLOAD_COLUMNS = ["mutation_class_id", "labeled_quivers", "boundary_quivers"]
_QUIVER_COLUMNS = [
    "id", "n", "exchange_matrix", "mutation_class_id", "max_edge",
    "is_acyclic", "is_connected", "is_bipartite", "is_abundant", "is_planar",
    "labeling_count", "representation_type", "symmetry_group",
]
_STATS_COLUMNS = ["n", "quiver_count", "labeled_quiver_count", "class_count"]


def render_rank_sql(n: int, rows: dict, *, bound: int) -> str:
    """
    Self-contained, idempotent SQL for one rank: replace every rank-n row.

    No explicit transaction statements (D1 rejects BEGIN/COMMIT inside a
    batch; `wrangler d1 execute --file` already applies the file as a batch).
    """
    parts = [
        f"-- Quiver Mutation Database — rank {n} "
        f"(bound |b_ij| <= {bound}).",
        "-- Generated by: python scripts/populate.py --export-d1 <dir>",
        "-- Import with:  wrangler d1 execute qmd --remote --file=<this file>",
        "-- Idempotent: deletes and re-inserts every rank-"
        f"{n} row, including its rank_stats aggregate.",
        "",
        "DELETE FROM mutation_class_payloads WHERE mutation_class_id IN "
        f"(SELECT id FROM mutation_classes WHERE n = {n});",
        f"DELETE FROM quivers WHERE n = {n};",
        f"DELETE FROM mutation_classes WHERE n = {n};",
        f"DELETE FROM rank_stats WHERE n = {n};",
        "",
    ]
    # Parents before children (quivers FK -> mutation_classes).
    parts.extend(_insert_stmts("mutation_classes", _MC_COLUMNS,
                               rows["mutation_classes"], chunk=40))
    # Payload rows carry whole labeled orbits — keep statements small.
    parts.extend(_insert_stmts("mutation_class_payloads", _PAYLOAD_COLUMNS,
                               rows["mutation_class_payloads"], chunk=5))
    parts.extend(_insert_stmts("quivers", _QUIVER_COLUMNS,
                               rows["quivers"], chunk=40))
    parts.extend(_insert_stmts("rank_stats", _STATS_COLUMNS,
                               [rows["rank_stats"]], chunk=1))
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Checkpointed per-rank driver
# ---------------------------------------------------------------------------

def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def export_ranks(out_dir: str, *, max_vertices: int, bound: int,
                 ranks: Optional[list[int]] = None, force: bool = False,
                 log=print) -> None:
    """
    Generate and export ranks 1..max_vertices (or the given `ranks`), one SQL
    file per rank, checkpointed so an interrupted run resumes where it left
    off.  A rank is skipped when the manifest already records it for the same
    bound and its file's hash still matches (pass force=True to redo).

    Ranks are processed in ascending order because the is_mutation_acyclic
    subquiver fallback consumes lower-rank results; each rank's quiver-level
    states are persisted to acyclicity-n{k}.json so a resumed run (or a later
    higher-rank run) can seed from them without regenerating.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    manifest = _load_json(manifest_path) or {"bound": bound, "ranks": {}}

    if manifest.get("bound") != bound:
        if not force:
            raise SystemExit(
                f"{manifest_path} was generated with bound="
                f"{manifest.get('bound')}, requested bound={bound}. "
                "Re-run with --force to regenerate everything."
            )
        manifest = {"bound": bound, "ranks": {}}

    todo = sorted(ranks) if ranks else list(range(1, max_vertices + 1))

    for n in todo:
        file_name = f"qmd-n{n}.sql"
        file_path = os.path.join(out_dir, file_name)
        entry = manifest["ranks"].get(str(n))

        if not force and entry and os.path.exists(file_path):
            with open(file_path, encoding="utf-8") as f:
                if _sha256(f.read()) == entry["sha256"]:
                    log(f"  rank {n}: up to date ({file_name}), skipping")
                    continue
            log(f"  rank {n}: {file_name} does not match manifest, regenerating")

        # Seed the subquiver fallback with every lower rank's checkpoint.
        known: dict = {}
        for j in range(1, n):
            ck = _load_json(os.path.join(out_dir, f"acyclicity-n{j}.json"))
            if ck is None:
                raise SystemExit(
                    f"rank {n} needs acyclicity-n{j}.json (lower-rank "
                    f"checkpoint) but it is missing — export rank {j} first "
                    "(or re-run without --ranks)."
                )
            known.update(ck)

        log(f"  rank {n}: generating (bound |b_ij| <= {bound}) ...")
        result = run_generation(max_vertices=n, bound=bound, ranks=[n])
        rows = build_rank_rows(result, n, known_acyclicity=known)
        sql = render_rank_sql(n, rows, bound=bound)

        _atomic_write(file_path, sql)
        _atomic_write(os.path.join(out_dir, f"acyclicity-n{n}.json"),
                      json.dumps(rows["acyclicity_by_qid"], sort_keys=True))
        manifest["ranks"][str(n)] = {
            "file": file_name,
            "sha256": _sha256(sql),
            "quiver_count": rows["rank_stats"]["quiver_count"],
            "labeled_quiver_count": rows["rank_stats"]["labeled_quiver_count"],
            "class_count": rows["rank_stats"]["class_count"],
        }
        _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
        log(f"  rank {n}: wrote {file_name} "
            f"({rows['rank_stats']['quiver_count']} quivers, "
            f"{rows['rank_stats']['class_count']} classes)")
