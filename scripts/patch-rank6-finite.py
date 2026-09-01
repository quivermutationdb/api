#!/usr/bin/env python3
"""
Supplemental patch for an already-exported rank: the mutation-finite fix.

`stage_label` runs at a small node cap because it visits every quiver in the
cell, and a class larger than that cap can never drain — so every
mutation-finite class above the cap came back *unknown*. Rank 6 was exported
that way: 492 quivers shipped with `mutation_finite` NULL, and ten of its
thirteen mutation-finite classes had no `mutation_classes` row at all. (The
three that made it, A6/D6/E6, were rescued only because the uniform sample
happened to land in all three.)

Regenerating the rank costs ~6 hours — a re-sample plus a 4-hour re-render of
6.8 GB of parts — to change about 700 rows, so this emits an additive patch
instead: UPDATEs for the affected quivers, complete rows and labelings for
every mutation-finite class, and a corrected `rank_stats`. The parts are
appended to the manifest, so `import-d1.sh` applies them after the base parts,
and re-applying them is idempotent.

Prerequisites — the scratch table must already hold the corrected values:
    stage_resolve(...)          # settles every unknown at a high cap
    stage_finite_classes(...)   # complete rows for every finite class

Usage:
    python scripts/patch-rank6-finite.py dist/d1 --rank 6
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmd import __version__ as PIPELINE_VERSION
from qmd.d1_export import (
    DEFAULT_PART_BYTES, _MC_COLUMNS, _LABELING_COLUMNS, _PartWriter,
    _insert_stmts, _labeling_rows, _lit, _load_json, _shards_config,
    _sha256_file, shard_keys_for, shard_of,
)


def _affected_quivers(con, settled: list[str]) -> dict:
    """
    Quiver rows the base export got wrong, with their corrected values.

    Two disjoint reasons a row is wrong, unioned:
      * it shipped `mutation_finite` NULL — every id the resolve stage settled;
      * it belongs to a mutation-finite class whose membership the base export
        never recorded (`mutation_class_id`, `labeling_count`).
    """
    ids = set(settled)
    # Unordered on purpose: ORDER BY on the TEXT primary key turns this
    # sequential scan into one random page seek per row (see CLAUDE.md).
    finite = con.execute(
        "SELECT id FROM quivers WHERE mutation_finite = 1").fetchall()
    ids |= {r[0] for r in finite}
    out = {}
    for qid in sorted(ids):
        row = con.execute(
            "SELECT mutation_finite, mutation_class_id, labeling_count "
            "FROM quivers WHERE id = ?", (qid,)).fetchone()
        if row is None:
            raise SystemExit(f"{qid} is not in the scratch table")
        out[qid] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    args = ap.parse_args()
    n, out_dir = args.rank, args.out_dir

    scratch = os.path.join(out_dir, f"work-n{n}.sqlite")
    con = sqlite3.connect(f"file:{scratch}?mode=ro", uri=True)
    con.execute("PRAGMA cache_size=-2000000")

    marker = con.execute("SELECT info FROM stages WHERE name='resolve'").fetchone()
    if not marker:
        raise SystemExit("the resolve stage has not run on this scratch database")
    settled = json.loads(marker[0]).get("settled", [])

    finite_pkl = os.path.join(out_dir, f"finite-n{n}.pkl")
    if not os.path.exists(finite_pkl):
        raise SystemExit(f"missing {finite_pkl} — run stage_finite_classes first")
    finite = pickle.load(open(finite_pkl, "rb"))
    class_rows = finite["mutation_classes"]
    classes = finite["classes"]

    quivers = _affected_quivers(con, settled)
    print(f"rank {n}: {len(settled)} unresolved + {len(quivers) - len(settled)} newly "
          f"classed = {len(quivers)} quiver row(s); {len(class_rows)} finite class(es)")

    cfg = _shards_config()
    parts: list[dict] = []
    for key, database in shard_keys_for(n, cfg):
        if key == "main":
            continue
        mc_here = [r for r in class_rows if shard_of(r["id"], n, cfg)[0] == key]
        q_here = {q: v for q, v in quivers.items() if shard_of(q, n, cfg)[0] == key}
        if not mc_here and not q_here:
            continue
        suffix = "s" + key.split(".")[1]
        w = _PartWriter(out_dir, f"qmd-n{n}.patch-{suffix}", args.part_bytes)
        w.write(f"-- Quiver Mutation Database — rank {n}, shard {key}: mutation-finite patch.")
        w.write(f"-- Pipeline {PIPELINE_VERSION}. Apply AFTER the base parts for this rank.")
        w.write("-- Idempotent: re-applying replaces the same rows with the same values.")

        if mc_here:
            ids = ", ".join(_lit(r["id"]) for r in mc_here)
            # labelings cascade off mutation_classes, and quivers.mutation_class_id
            # is ON DELETE SET NULL — so classes go first, then the quiver UPDATEs
            # below restore membership, then the labelings are re-inserted.
            w.write(f"DELETE FROM labelings WHERE mutation_class_id IN ({ids});")
            w.write(f"DELETE FROM mutation_classes WHERE id IN ({ids});")
            for stmt in _insert_stmts("mutation_classes", _MC_COLUMNS, mc_here):
                w.write(stmt)

        for qid, (fin, mc_id, lab) in sorted(q_here.items()):
            w.write(
                f"UPDATE quivers SET mutation_finite = {_lit(fin)}, "
                f"mutation_class_id = {_lit(mc_id)}, labeling_count = {_lit(lab)} "
                f"WHERE id = {_lit(qid)};")

        if mc_here:
            keys = {r["id"] for r in mc_here}
            for stmt in _insert_stmts("labelings", _LABELING_COLUMNS,
                                      _labeling_rows(classes, keys)):
                w.write(stmt)

        got = w.close()
        parts.extend({**p, "shard": key, "database": database} for p in got)
        print(f"  shard {key}: {len(mc_here)} class(es), {len(q_here)} quiver update(s), "
              f"{len(got)} part(s)")

    # rank_stats lives in main and is a whole-rank aggregate.
    labeled = sum(r["class_size"] or 0 for r in class_rows)
    w = _PartWriter(out_dir, f"qmd-n{n}.patch-stats", args.part_bytes)
    w.write(f"-- rank {n}: rank_stats corrected for the mutation-finite patch.")
    w.write(f"UPDATE rank_stats SET class_count = "
            f"(SELECT count(*) FROM mutation_classes WHERE n = {n}) WHERE n = {n};")
    w.write(f"UPDATE rank_stats SET labeled_quiver_count = "
            f"(SELECT coalesce(sum(class_size), 0) FROM mutation_classes WHERE n = {n}) "
            f"WHERE n = {n};")
    got = w.close()
    parts.extend({**p, "shard": "main", "database": cfg["main"]["database"]} for p in got)
    print(f"  main: rank_stats, {len(got)} part(s)")
    print(f"  finite labeled matrices: {labeled}")

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = _load_json(manifest_path)
    if manifest is None:
        raise SystemExit(f"no manifest at {manifest_path}")
    entry = manifest["ranks"][str(n)]
    # Drop any earlier run of this patch, then append — the base parts keep
    # their positions, so the patch always lands last.
    entry["parts"] = [p for p in entry["parts"] if "patch-" not in p["file"]]
    entry["parts"].extend(parts)
    entry["patch"] = {
        "reason": "mutation-finite classes missing and unknowns unresolved",
        "quiver_updates": len(quivers),
        "finite_classes": len(class_rows),
        "finite_labeled_quiver_count": labeled,
        "pipeline_version": PIPELINE_VERSION,
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, manifest_path)
    print(f"manifest: {len(parts)} patch part(s) appended to rank {n}")


if __name__ == "__main__":
    main()
