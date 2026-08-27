"""
qmd/bigcell.py — streaming pipeline for a cell too large to hold in memory
(rank 6 at |b_ij| <= 2: 42.5 M connected quivers; docs/PHASE3.md §1, plan (c)).

Nothing about the census is loaded whole. A scratch SQLite database
(`<out_dir>/work-n{k}.sqlite`) holds one row per quiver, filled stage by
stage; each stage is resumable and parallel:

  1. generate   orderly generation of level k from the (k-1) level (kept in
                memory: 82,880 rank-5 parents), children streamed into the
                scratch table in id order (connected only)
  2. invariants per-quiver invariants (parallel, chunked)
  3. label      a capped unlabeled BFS from EVERY quiver at bound 2: a wall
                crossing proves the whole explored set mutation-infinite
                (Derksen–Owen), which is written back to every quiver seen —
                so most quivers get mutation_finite = 0 without a class row;
                a class that drains proves finite for all its members
  4. sample     class rows for a uniform sample of K quivers (their full
                capped explorations, glued, with invariants) via the normal
                run_generation path
  5. export     per-shard part files straight from the scratch table

The output is the same manifest format as qmd/d1_export.export_ranks.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import random
import sqlite3
from typing import Iterator, Optional

from qmd import __version__ as PIPELINE_VERSION
from qmd import census, invariants
from qmd.core import (
    _bfs_unlabeled, canonical_form, is_acyclic, is_connected, max_edge, quiver_id, run_generation,
)
from qmd.d1_export import (
    EXPLORE_BOUND, _atomic_write, _insert_stmts, _header, _load_json, _lit, _shard_counts,
    _shards_config, _sha256_file, build_rank_rows, shard_keys_for, shard_of,
    _MC_COLUMNS, _LABELING_COLUMNS, _QUIVER_COLUMNS, _STATS_COLUMNS, _PartWriter,
    DEFAULT_PART_BYTES, _labeling_rows,
)
from qmd.encoding import decode_upper, encode_upper

SCHEMA = """
CREATE TABLE IF NOT EXISTS quivers (
  id TEXT PRIMARY KEY, upper TEXT NOT NULL,
  max_edge INTEGER, is_acyclic INTEGER, is_connected INTEGER, is_bipartite INTEGER,
  is_abundant INTEGER, is_planar INTEGER, representation_type TEXT, symmetry_group TEXT,
  mutation_finite INTEGER, mutation_class_id TEXT, labeling_count INTEGER,
  invariants_done INTEGER DEFAULT 0, label_done INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS stages (name TEXT PRIMARY KEY, done INTEGER, info TEXT);
"""


def _db(path: str) -> sqlite3.Connection:
    """One connection per thread: the pool feeds task generators from its own
    thread, so generators open their own reader connection (WAL mode)."""
    con = sqlite3.connect(path, timeout=600)
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _stage_done(con, name) -> bool:
    r = con.execute("SELECT done FROM stages WHERE name=?", (name,)).fetchone()
    return bool(r and r[0])


def _mark(con, name, info=None) -> None:
    con.execute("INSERT OR REPLACE INTO stages VALUES (?, 1, ?)", (name, json.dumps(info or {})))
    con.commit()


# ---------------------------------------------------------------------------
# 1. generate
# ---------------------------------------------------------------------------

def _children_job(args):
    parent, h = args
    out = []
    for child in census.children(parent, h):
        if is_connected(child):
            cf = canonical_form(child)
            out.append((quiver_id(cf), encode_upper(cf)))
    return out


def stage_generate(con, n: int, h: int, workers: int, log) -> None:
    if _stage_done(con, "generate"):
        log("  generate: done"); return
    parents = list(census.generate_cell(n - 1, h, workers=workers,
                                        progress=lambda k, c: log(f"    level {k}: {c}")))
    log(f"    {len(parents)} parents at rank {n - 1}; extending ...")
    import multiprocessing as mp
    con.execute("DELETE FROM quivers")
    count = 0
    with mp.get_context("fork").Pool(workers) as pool:
        for i, kids in enumerate(pool.imap_unordered(_children_job, [(p, h) for p in parents], chunksize=4), 1):
            con.executemany("INSERT OR IGNORE INTO quivers (id, upper) VALUES (?, ?)", kids)
            count += len(kids)
            if i % 500 == 0 or i == len(parents):
                con.commit(); log(f"    extended {i}/{len(parents)} parents, {count} quivers")
    con.commit()
    total = con.execute("SELECT count(*) FROM quivers").fetchone()[0]
    expected = census.count_connected_quivers(n, h)
    if total != expected:
        raise SystemExit(f"generated {total} connected quivers, expected {expected}")
    _mark(con, "generate", {"quivers": total})
    log(f"  generate: {total} quivers (matches the exact count)")


# ---------------------------------------------------------------------------
# 2. invariants
# ---------------------------------------------------------------------------

def _inv_job(args):
    n, rows = args
    out = []
    for qid, upper in rows:
        m = decode_upper(n, upper)
        qi = invariants.quiver_invariants(m)
        out.append((max_edge(m), int(is_acyclic(m)), int(is_connected(m)), qi["is_bipartite"],
                    qi["is_abundant"], qi["is_planar"], qi["representation_type"],
                    json.dumps(qi["symmetry_group"], separators=(",", ":")), qid))
    return out


def stage_invariants(con, path: str, n: int, workers: int, log, chunk: int = 2000) -> None:
    if _stage_done(con, "invariants"):
        log("  invariants: done"); return
    todo = con.execute("SELECT count(*) FROM quivers WHERE invariants_done = 0").fetchone()[0]
    log(f"    invariants for {todo} quivers ...")
    import multiprocessing as mp

    def batches():
        rcon = sqlite3.connect(path, timeout=600)        # generator thread's own reader
        last = ""
        while True:
            rows = rcon.execute("SELECT id, upper FROM quivers WHERE invariants_done = 0 AND id > ? ORDER BY id LIMIT ?",
                                (last, chunk)).fetchall()
            if not rows:
                rcon.close()
                return
            last = rows[-1][0]
            yield (n, rows)

    done = 0
    with mp.get_context("fork").Pool(workers) as pool:
        for out in pool.imap_unordered(_inv_job, batches()):
            con.executemany("UPDATE quivers SET max_edge=?, is_acyclic=?, is_connected=?, is_bipartite=?, "
                            "is_abundant=?, is_planar=?, representation_type=?, symmetry_group=?, "
                            "invariants_done=1 WHERE id=?", out)
            con.commit()
            done += len(out)
            if done % (chunk * 25) == 0:
                log(f"    invariants {done}/{todo}")
    _mark(con, "invariants")
    log("  invariants: done")


# ---------------------------------------------------------------------------
# 3. label (finiteness for every quiver via capped unlabeled BFS)
# ---------------------------------------------------------------------------

def _label_job(args):
    n, cap, rows = args
    out = []
    for qid, upper in rows:
        orbit = _bfs_unlabeled(decode_upper(n, upper), EXPLORE_BOUND, cap)
        if orbit.crossed:                      # Derksen–Owen: the whole explored set is infinite
            out.append((0, sorted(orbit.qid_set)))
        elif not orbit.is_open:                # drained under the cap: finite, for every member
            out.append((1, sorted(orbit.qid_set)))
        else:                                  # truncated without a crossing: unknown
            out.append((None, [qid]))
    return out


def stage_label(con, path: str, n: int, cap: int, workers: int, log, chunk: int = 200) -> None:
    if _stage_done(con, "label"):
        log("  label: done"); return
    import multiprocessing as mp
    todo = con.execute("SELECT count(*) FROM quivers WHERE label_done = 0").fetchone()[0]
    log(f"    labelling finiteness for {todo} quivers (cap {cap}) ...")

    def batches():
        rcon = sqlite3.connect(path, timeout=600)        # generator thread's own connection
        while True:
            # Quivers already labelled by a neighbour's exploration are skipped.
            rows = rcon.execute("SELECT id, upper FROM quivers WHERE label_done = 0 ORDER BY id LIMIT ?", (chunk * workers,)).fetchall()
            if not rows:
                rcon.close()
                return
            for i in range(0, len(rows), chunk):
                yield (n, cap, rows[i:i + chunk])
            rcon.execute("UPDATE quivers SET label_done = 1 WHERE id IN (SELECT id FROM quivers WHERE label_done = 0 ORDER BY id LIMIT ?)", (len(rows),))
            rcon.commit()

    done = 0
    with mp.get_context("fork").Pool(workers) as pool:
        for out in pool.imap_unordered(_label_job, batches()):
            for value, qids in out:
                if value is not None:
                    con.executemany("UPDATE quivers SET mutation_finite = ?, label_done = 1 WHERE id = ? AND mutation_finite IS NULL",
                                    [(value, q) for q in qids])
            con.commit()
            done += len(out)
            if done % (chunk * 50) == 0:
                log(f"    labelled {done}/{todo}")
    _mark(con, "label")
    counts = con.execute("SELECT mutation_finite, count(*) FROM quivers GROUP BY 1").fetchall()
    log(f"  label: done {dict((k if k is not None else 'unknown', v) for k, v in counts)}")


# ---------------------------------------------------------------------------
# 4. sample: class rows for K quivers via the normal pipeline
# ---------------------------------------------------------------------------

def stage_sample(con, n: int, k: int, node_cap: int, workers: int, seed: int, la_timeout, known, log) -> dict:
    total = con.execute("SELECT count(*) FROM quivers").fetchone()[0]
    rng = random.Random(seed)
    # Uniform sample of rowids (ids are in insertion = arbitrary parent order; sample by rowid).
    picks = sorted(rng.sample(range(1, total + 1), min(k, total)))
    seeds = []
    for i in range(0, len(picks), 900):
        rows = con.execute(f"SELECT upper FROM quivers WHERE rowid IN ({','.join('?' * len(picks[i:i+900]))})", picks[i:i + 900]).fetchall()
        seeds.extend(decode_upper(n, r[0]) for r in rows)
    log(f"    exploring classes for {len(seeds)} sampled quivers (cap {node_cap}) ...")

    def prog(stage, i, tot):
        if i == tot or i % max(1, tot // 10) == 0:
            log(f"    {stage}: {i}/{tot}")
    result = run_generation(max_vertices=n, bound=EXPLORE_BOUND, ranks=[n], node_cap=node_cap,
                            seeds=seeds, workers=workers, progress=prog)
    log(f"    {len(result.quivers)} quivers in {len(result.classes)} classes; class invariants ...")
    rows = build_rank_rows(result, n, known_acyclicity=known, bound=2, node_cap=node_cap,
                           generator="orderly", census_size=None, la_timeout=la_timeout,
                           workers=workers, progress=prog)
    # Write class membership + finiteness back to the scratch table.
    for q in rows["quivers"]:
        con.execute("UPDATE quivers SET mutation_class_id=?, labeling_count=?, mutation_finite=coalesce(mutation_finite, ?) WHERE id=?",
                    (q["mutation_class_id"], q["labeling_count"], q["mutation_finite"], q["id"]))
    con.commit()
    return rows


# ---------------------------------------------------------------------------
# 5. export
# ---------------------------------------------------------------------------

def _quiver_rows(con, n: int) -> Iterator[dict]:
    cur = con.execute("SELECT id, upper, mutation_class_id, mutation_finite, max_edge, is_acyclic, is_connected, "
                      "is_bipartite, is_abundant, is_planar, labeling_count, representation_type, symmetry_group "
                      "FROM quivers ORDER BY id")
    for r in cur:
        yield {
            "id": r[0], "n": n, "exchange_matrix": r[1], "mutation_class_id": r[2],
            "mutation_finite": None if r[3] is None else bool(r[3]), "max_edge": r[4],
            "is_acyclic": bool(r[5]), "is_connected": bool(r[6]),
            "is_bipartite": None if r[7] is None else bool(r[7]),
            "is_abundant": None if r[8] is None else bool(r[8]),
            "is_planar": None if r[9] is None else bool(r[9]),
            "labeling_count": r[10], "representation_type": r[11],
            "symmetry_group": json.loads(r[12]) if r[12] else None,
        }


def stage_export(con, out_dir: str, n: int, h: int, class_rows: dict, node_cap: int,
                 sample_k: int, part_bytes: int, log) -> list[dict]:
    cfg = _shards_config()
    total = con.execute("SELECT count(*) FROM quivers").fetchone()[0]
    parts: list[dict] = []
    for key, database in shard_keys_for(n, cfg):
        suffix = "main" if key == "main" else "s" + key.split(".")[1]
        w = _PartWriter(out_dir, f"qmd-n{n}.{suffix}", part_bytes)
        for stmt in _header(n, h, key):
            w.write(stmt)
        w.write(f"DELETE FROM labelings WHERE mutation_class_id IN (SELECT id FROM mutation_classes WHERE n = {n});")
        w.write(f"DELETE FROM quivers WHERE n = {n};")
        w.write(f"DELETE FROM mutation_classes WHERE n = {n};")
        mc_ids = {r["id"] for r in class_rows["mutation_classes"] if shard_of(r["id"], n, cfg)[0] == key}
        for stmt in _insert_stmts("mutation_classes", _MC_COLUMNS,
                                  (r for r in class_rows["mutation_classes"] if r["id"] in mc_ids)):
            w.write(stmt)
        for stmt in _insert_stmts("quivers", _QUIVER_COLUMNS,
                                  (r for r in _quiver_rows(con, n) if shard_of(r["id"], n, cfg)[0] == key)):
            w.write(stmt)
        for stmt in _insert_stmts("labelings", _LABELING_COLUMNS, _labeling_rows(class_rows["classes"], mc_ids)):
            w.write(stmt)
        parts.extend({**p, "shard": key, "database": database} for p in w.close())
        log(f"    shard {key}: {len(parts)} part(s) so far")
    # rank_stats (main)
    shard_counts = {key: {"quivers": 0, "classes": 0} for key, _ in shard_keys_for(n, cfg)}
    for (qid,) in con.execute("SELECT id FROM quivers"):
        shard_counts[shard_of(qid, n, cfg)[0]]["quivers"] += 1
    for r in class_rows["mutation_classes"]:
        shard_counts[shard_of(r["id"], n, cfg)[0]]["classes"] += 1
    stats = {
        "n": n, "quiver_count": total,
        "labeled_quiver_count": class_rows["rank_stats"]["labeled_quiver_count"],
        "class_count": len(class_rows["mutation_classes"]),
        "bound": h, "node_cap": node_cap,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_version": PIPELINE_VERSION,
        "generator": f"orderly; classes for sample:{sample_k}",
        "census_size": census.count_connected_quivers(n, h),
        "shard_counts": shard_counts,
    }
    w = _PartWriter(out_dir, f"qmd-n{n}.stats", part_bytes)
    w.write(f"DELETE FROM rank_stats WHERE n = {n};")
    for stmt in _insert_stmts("rank_stats", _STATS_COLUMNS, [stats]):
        w.write(stmt)
    parts.extend({**p, "shard": "main", "database": cfg["main"]["database"]} for p in w.close())
    return parts


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def export_big_cell(out_dir: str, *, n: int, h: int, label_cap: int = 20, node_cap: int = 100,
                    sample: int = 1_000_000, sample_seed: int = 0, workers: int = 8,
                    la_timeout: Optional[float] = 1.0, part_bytes: int = DEFAULT_PART_BYTES,
                    log=print) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"work-n{n}.sqlite")
    con = _db(path)
    log(f"  rank {n} (cell |b_ij| <= {h}): streaming pipeline in {path}")
    stage_generate(con, n, h, workers, log)
    stage_invariants(con, path, n, workers, log)
    stage_label(con, path, n, label_cap, workers, log)

    known: dict = {}
    for j in range(1, n):
        ck = _load_json(os.path.join(out_dir, f"acyclicity-n{j}.json"))
        if ck is None:
            raise SystemExit(f"rank {n} needs acyclicity-n{j}.json — export the lower ranks first")
        known.update(ck)
    class_rows = stage_sample(con, n, sample, node_cap, workers, sample_seed, la_timeout, known, log)
    parts = stage_export(con, out_dir, n, h, class_rows, node_cap, sample, part_bytes, log)
    _atomic_write(os.path.join(out_dir, f"acyclicity-n{n}.json"),
                  json.dumps(class_rows["acyclicity_by_qid"], sort_keys=True))

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = _load_json(manifest_path) or {"ranks": {}}
    manifest.setdefault("ranks", {})["pipeline_version"] = PIPELINE_VERSION
    manifest["ranks"][str(n)] = {
        "parts": parts,
        "depends_on": {f"acyclicity-n{j}.json": _sha256_file(os.path.join(out_dir, f"acyclicity-n{j}.json")) for j in range(1, n)},
        "settings": {"bound": h, "node_cap": node_cap, "label_cap": label_cap, "generator": "bigcell",
                     "sample": sample, "la_timeout": la_timeout, "schema": 3},
        "quiver_count": con.execute("SELECT count(*) FROM quivers").fetchone()[0],
        "class_count": len(class_rows["mutation_classes"]),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    log(f"  rank {n}: wrote {len(parts)} part(s), {sum(p['bytes'] for p in parts) / 1e9:.2f} GB")
