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
  3b. resolve   re-explore whatever the capped label pass left unknown, at a
                cap high enough to settle it (a class bigger than the label
                cap can never drain, so every finite class above it comes back
                unknown); hundreds of rows, seconds of work
  4. sample     class rows for a uniform sample of K quivers (their full
                capped explorations, glued, with invariants) via the normal
                run_generation path, PLUS a complete exploration of every
                mutation-finite class — sampling cannot be trusted to find
                those (rank 6 has 428 finite quivers out of 42.5 M)
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
    DEFAULT_PART_BYTES, _labeling_rows, _curated_seeds,
)
from qmd.encoding import decode_upper, encode_upper

# Page cache for every scratch connection, in KiB (2 GB). The label stage
# seeks random primary keys across a 7 GB table; without this it is disk-bound.
CACHE_KIB = int(os.environ.get("QMD_SQLITE_CACHE_KIB", 2_000_000))

SCHEMA = """
CREATE TABLE IF NOT EXISTS quivers (
  id TEXT PRIMARY KEY, upper TEXT NOT NULL,
  max_edge INTEGER, is_acyclic INTEGER, is_connected INTEGER, is_bipartite INTEGER,
  is_abundant INTEGER, is_planar INTEGER, representation_type TEXT, symmetry_group TEXT,
  mutation_finite INTEGER, mutation_class_id TEXT, labeling_count INTEGER,
  invariants_done INTEGER DEFAULT 0, label_done INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS stages (name TEXT PRIMARY KEY, done INTEGER, info TEXT);
-- How far a stage's id-ordered walk has committed. The label stage resumes
-- from here instead of re-walking (and re-skipping) the settled prefix.
CREATE TABLE IF NOT EXISTS watermarks (stage TEXT PRIMARY KEY, last_id TEXT NOT NULL);
-- Verdicts are APPENDED here, never written into `quivers` row by row: a
-- scattered `UPDATE quivers ... WHERE id = ?` costs one random page read per
-- row against a multi-GB table, which is what stalled the first attempt.
-- stage_label_apply folds this table into `quivers` in a single id-ordered pass.
CREATE TABLE IF NOT EXISTS label_verdicts (id TEXT PRIMARY KEY, value INTEGER);
-- Parents whose extension has been committed: the generate stage resumes from
-- here, so a kill costs at most one commit batch instead of the whole stage.
CREATE TABLE IF NOT EXISTS parents_done (idx INTEGER PRIMARY KEY);
"""


def _log(*args) -> None:
    """Default progress sink. `flush` is not optional: run-detached.sh sends
    stdout to a file, so Python block-buffers it, and a stage that prints a line
    every few minutes had its progress sitting in an unflushed buffer — lost on
    every kill. That is why the first rank-6 label run looked like it had
    produced nothing when it had in fact settled ~2.9 M quivers."""
    print(*args, flush=True)


def _db(path: str) -> sqlite3.Connection:
    """One connection per thread: the pool feeds task generators from its own
    thread, so generators open their own reader connection (WAL mode)."""
    con = sqlite3.connect(path, timeout=600)
    con.executescript(SCHEMA)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    # A 2 MB default page cache against a multi-GB table means one synchronous
    # disk read per random primary-key seek. Give it room (negative = KiB).
    con.execute(f"PRAGMA cache_size=-{CACHE_KIB}")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA mmap_size=%d" % (8 << 30))
    return con


def _tri(v) -> Optional[int]:
    """Three-state boolean to SQLite: 1 / 0 / NULL (unknown)."""
    return None if v is None else int(v)


def _stage_done(con, name) -> bool:
    r = con.execute("SELECT done FROM stages WHERE name=?", (name,)).fetchone()
    return bool(r and r[0])


def _mark(con, name, info=None) -> None:
    con.execute("INSERT OR REPLACE INTO stages VALUES (?, 1, ?)", (name, json.dumps(info or {})))
    con.commit()


def _watermark(con, stage: str) -> str:
    r = con.execute("SELECT last_id FROM watermarks WHERE stage=?", (stage,)).fetchone()
    return r[0] if r else ""


def _set_watermark(con, stage: str, last_id: str) -> None:
    con.execute("INSERT OR REPLACE INTO watermarks VALUES (?, ?)", (stage, last_id))


# ---------------------------------------------------------------------------
# 1. generate
# ---------------------------------------------------------------------------

def _children_job(args):
    idx, parent, h = args
    out = []
    for child in census.children(parent, h):
        if is_connected(child):
            cf = canonical_form(child)
            out.append((quiver_id(cf), encode_upper(cf)))
    return idx, out


BATCH_PARENTS = 200          # commit (and checkpoint) this often while extending


def stage_generate(con, n: int, h: int, workers: int, log) -> None:
    """
    Extend every key-minimal (n-1)-parent into rank-n children, resumably.

    Parents are deterministic and sorted, so their index identifies them across
    runs; `parents_done` records the ones whose children are committed. A kill
    therefore costs at most BATCH_PARENTS parents of work, not the stage.
    Children are INSERT OR IGNORE, so re-running a parent is harmless.
    """
    if _stage_done(con, "generate"):
        log("  generate: done"); return
    parents = list(census.generate_cell(n - 1, h, workers=workers,
                                        progress=lambda k, c: log(f"    level {k}: {c}")))
    done = {r[0] for r in con.execute("SELECT idx FROM parents_done")}
    todo = [(i, p, h) for i, p in enumerate(parents) if i not in done]
    have = con.execute("SELECT count(*) FROM quivers").fetchone()[0]
    log(f"    {len(parents)} parents at rank {n - 1}; {len(done)} already extended "
        f"({have} quivers stored), {len(todo)} to go ...")
    import multiprocessing as mp
    count = have
    pending = 0
    with mp.get_context("fork").Pool(workers) as pool:
        for i, (idx, kids) in enumerate(pool.imap_unordered(_children_job, todo, chunksize=4), 1):
            con.executemany("INSERT OR IGNORE INTO quivers (id, upper) VALUES (?, ?)", kids)
            con.execute("INSERT OR IGNORE INTO parents_done VALUES (?)", (idx,))
            count += len(kids)
            pending += 1
            if pending >= BATCH_PARENTS or i == len(todo):
                con.commit()
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")   # keep the WAL bounded
                pending = 0
                log(f"    extended {len(done) + i}/{len(parents)} parents, {count} quivers")
    con.commit()
    total = con.execute("SELECT count(*) FROM quivers").fetchone()[0]
    expected = census.count_connected_quivers(n, h)
    if total != expected:
        raise SystemExit(f"generated {total} connected quivers, expected {expected} "
                         f"(resume by re-running; parents_done has {len(done) + len(todo)} entries)")
    _mark(con, "generate", {"quivers": total})
    log(f"  generate: {total} quivers (matches the exact count)")


def stage_generate_sample(con, n: int, h: int, k: int, seed: int, log) -> None:
    """
    Seed the scratch table with a uniform sample of the cell, for cells that
    cannot be enumerated: (8,1) holds 572,849,763 connected quivers.

    The sample is drawn over labeled matrices and then canonicalised, so
    symmetric quivers are under-represented exactly as they are among labeled
    matrices (`qmd.census.sample_cell`) — document that with any ML dataset.
    The mutation-finite classes are NOT left to this draw: they are seeded by
    construction in `finite_class_seeds`.
    """
    if _stage_done(con, "generate"):
        log("  generate: done")
        return
    exact = census.count_connected_quivers(n, h)
    log(f"    sampling {k:,} of {exact:,} connected quivers in the cell ...")
    picks = census.sample_cell(n, h, k, seed=seed, connected_only=True)
    con.executemany("INSERT OR IGNORE INTO quivers (id, upper) VALUES (?, ?)",
                    [(quiver_id(m), encode_upper(m)) for m in picks])
    con.commit()
    total = con.execute("SELECT count(*) FROM quivers").fetchone()[0]
    _mark(con, "generate", {"quivers": total, "cell_sample": k})
    log(f"  generate: {total} sampled quivers (cell {exact:,})")


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
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                log(f"    invariants {done}/{todo}")
    _mark(con, "invariants")
    log("  invariants: done")


# ---------------------------------------------------------------------------
# 3. label (finiteness for every quiver via capped unlabeled BFS)
# ---------------------------------------------------------------------------

def _label_job(args):
    """`seq` is echoed back so the caller can advance its watermark only past
    batches whose verdicts it has committed (imap_unordered completes out of
    order)."""
    seq, n, cap, rows = args
    out = []
    for qid, upper in rows:
        orbit = _bfs_unlabeled(decode_upper(n, upper), EXPLORE_BOUND, cap)
        if orbit.crossed:                      # Derksen–Owen: the whole explored set is infinite
            out.append((0, sorted(orbit.qid_set)))
        elif not orbit.is_open:                # drained under the cap: finite, for every member
            out.append((1, sorted(orbit.qid_set)))
        else:                                  # truncated without a crossing: unknown
            out.append((None, [qid]))
    return seq, out


def _seed_label_watermark(con, path: str, log, chunk: int = 50_000) -> str:
    """
    Advance the label watermark over the already-settled prefix, committing as
    it goes, and return where the real work starts.

    The first attempt at this stage settled a contiguous prefix of ~4.5 M rows
    and recorded no position, so every restart re-walked it (~13 minutes) before
    finding anything to do. Walking it once and committing the watermark makes
    that cost one-time and interruptible. Rows settled out of order by a
    neighbour's exploration are simply skipped again later — the watermark only
    ever moves over a run of fully settled rows.
    """
    last = _watermark(con, "label")
    moved = 0
    while True:
        rows = con.execute(
            "SELECT id, mutation_finite, label_done FROM quivers WHERE id > ? ORDER BY id LIMIT ?",
            (last, chunk)).fetchall()
        if not rows:
            break
        stop = None
        for qid, known, done in rows:
            if known is None and not done:
                stop = qid
                break
            last = qid
            moved += 1
        if last != _watermark(con, "label"):
            _set_watermark(con, "label", last)
            con.commit()
        if stop is not None:
            break
        if moved % 1_000_000 < chunk:
            log(f"    skipped {moved} already-settled rows (id <= {last})")
    if moved:
        log(f"    resume: skipped {moved} settled rows, work starts after {last!r}")
    return last


def stage_label(con, path: str, n: int, cap: int, workers: int, log, chunk: int = 200,
                log_every: int = 20_000) -> None:
    """
    Give every quiver a finiteness verdict with a capped unlabeled BFS.

    A crossing of the weight bound proves the whole explored set infinite
    (Derksen–Owen) and a drained search proves it finite, so ONE exploration
    usually settles up to `cap` quivers at once — the seeds are walked in id
    order and any quiver already settled is skipped.

    Two rules keep this affordable at 42 M rows, both learned the hard way:

    * The walk resumes from a committed **watermark**, not from a scan for
      unsettled rows. Re-scanning from the start re-skips every settled row on
      every batch (quadratic); walking the settled prefix again on each restart
      cost 13 minutes before this.
    * Verdicts are **appended** to `label_verdicts`, never written into
      `quivers` row by row. A scattered `UPDATE quivers ... WHERE id = ?` costs
      one random page read per row against a 7 GB table — profiling the first
      attempt found 90 % of the writer's time inside that seek, at a rate that
      would have taken weeks. `stage_label_apply` folds the verdicts in one
      id-ordered pass, which traverses the B-tree sequentially instead.

    `mutation_finite` is the only ledger of what is settled, so an interrupted
    run never leaves a quiver marked-but-unresolved, and the verdicts already
    on disk are inherited by the next run.
    """
    if _stage_done(con, "label"):
        log("  label: done"); return
    import multiprocessing as mp
    total = con.execute("SELECT count(*) FROM quivers").fetchone()[0]
    # Fold in anything a killed run left staged, so the skip test below sees it.
    stage_label_apply(con, log)
    start = _seed_label_watermark(con, path, log)
    log(f"    labelling finiteness of {total} quivers (cap {cap}) from id > {start!r} ...")

    # Ids of the quivers each yielded batch covers, so the watermark only
    # advances past a batch whose verdicts have been committed.
    covers: dict[int, str] = {}

    def batches():
        rcon = _db(path)
        last = start
        pending: list[tuple] = []
        seq = 0
        while True:
            rows = rcon.execute(
                "SELECT id, upper, mutation_finite, label_done FROM quivers "
                "WHERE id > ? ORDER BY id LIMIT ?",
                (last, chunk * workers * 4)).fetchall()
            if not rows:
                break
            last = rows[-1][0]
            for qid, upper, known, done_row in rows:
                # Settled here or by a neighbour's exploration — or explored and
                # left *unknown* (truncated, no crossing), which re-exploring at
                # the same cap cannot improve. Same test as _seed_label_watermark.
                if known is not None or done_row:
                    continue
                pending.append((qid, upper))
                if len(pending) == chunk:
                    seq += 1
                    covers[seq] = pending[-1][0]
                    yield (seq, n, cap, pending)
                    pending = []
        if pending:
            seq += 1
            covers[seq] = pending[-1][0]
            yield (seq, n, cap, pending)
        rcon.close()

    done = 0
    since_log = 0
    high = start
    with mp.get_context("fork").Pool(workers) as pool:
        for seq, out in pool.imap_unordered(_label_job, batches()):
            for value, qids in out:
                con.executemany("INSERT OR IGNORE INTO label_verdicts VALUES (?, ?)",
                                [(q, value) for q in qids])
                done += len(qids)
                since_log += len(qids)
            # imap_unordered may complete out of order; only advance the
            # watermark monotonically, so at worst a batch is recomputed.
            edge = covers.pop(seq, None)
            if edge is not None and edge > high:
                high = edge
                _set_watermark(con, "label", high)
            con.commit()
            if since_log >= log_every:
                # Draining here keeps the staging table small and `quivers` up
                # to date, so the walk's skip test stays effective.
                stage_label_apply(con, log)
                log(f"    labelled {done} this run of {total} total (id <= {high})")
                since_log = 0
    con.commit()
    stage_label_apply(con, log)
    _mark(con, "label")
    counts = con.execute("SELECT mutation_finite, count(*) FROM quivers GROUP BY 1").fetchall()
    log(f"  label: done {dict((k if k is not None else 'unknown', v) for k, v in counts)}")


def stage_label_apply(con, log, chunk: int = 200_000) -> int:
    """
    Drain `label_verdicts` into `quivers.mutation_finite` in one id-ordered pass.

    Both tables are walked in primary-key order, so writes hit the target
    B-tree sequentially instead of seeking once per row — the whole reason
    verdicts are staged rather than written where they are produced.

    Applied rows are deleted in the same transaction, so the staging table
    stays small and the pass is cheap enough to run periodically and on resume.
    `mutation_finite IS NULL` guards the write: a truncated search without a
    crossing records its seed as *unknown* (value NULL), and unknown must never
    overwrite a verdict some other exploration proved.
    """
    applied = 0
    while True:
        rows = con.execute("SELECT id, value FROM label_verdicts ORDER BY id LIMIT ?",
                           (chunk,)).fetchall()
        if not rows:
            break
        known = [(v, q) for q, v in rows if v is not None]
        if known:
            con.executemany("UPDATE quivers SET mutation_finite = ?, label_done = 1 "
                            "WHERE id = ? AND mutation_finite IS NULL", known)
        con.executemany("UPDATE quivers SET label_done = 1 WHERE id = ?",
                        [(q,) for q, _ in rows])
        con.executemany("DELETE FROM label_verdicts WHERE id = ?", [(q,) for q, _ in rows])
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        applied += len(rows)
    if applied:
        log(f"    applied {applied} staged verdicts")
    return applied


# ---------------------------------------------------------------------------
# 3b. resolve: settle every quiver the capped label pass left unknown
# ---------------------------------------------------------------------------

RESOLVE_CAP = 100_000


def stage_resolve(con, n: int, cap: int, workers: int, log, chunk: int = 50) -> None:
    """
    Re-explore every quiver still marked *unknown*, at a cap high enough to
    settle it.

    `stage_label` has to run at a small cap because it visits all 42 M rows,
    and a class larger than that cap can never drain — so every mutation-finite
    class bigger than the cap comes back unknown. At rank 6 that silently
    swallowed A6, D6 and E6 (49, 80 and 67 quivers); they were only rescued
    because the random sample happened to land in all three, a ~3% event.

    The leftovers are few — hundreds, not millions — so a second pass at a cap
    three orders of magnitude larger costs seconds and leaves nothing
    undecided. The slowest rank-6 quiver needed 1,089 visits before it crossed
    the wall, which is why the cap here is not merely "a bit bigger".

    Verdicts go through `label_verdicts` and the same id-ordered apply pass as
    the label stage; nothing is written to the big table row by row.
    """
    if _stage_done(con, "resolve"):
        return
    # No ORDER BY: on a TEXT PRIMARY KEY that would walk the implicit index and
    # seek the table once per row — 42 M random page reads. An unordered scan is
    # sequential, and the handful of survivors are sorted in Python.
    pending = sorted(con.execute(
        "SELECT id, upper FROM quivers WHERE mutation_finite IS NULL").fetchall())
    log(f"    resolving {len(pending)} unknown quivers (cap {cap}) ...")
    if pending:
        import multiprocessing as mp
        batches = [(i, n, cap, pending[i:i + chunk]) for i in range(0, len(pending), chunk)]
        settled = 0
        with mp.get_context("fork").Pool(workers) as pool:
            for _seq, out in pool.imap_unordered(_label_job, batches):
                for value, qids in out:
                    con.executemany("INSERT OR IGNORE INTO label_verdicts VALUES (?, ?)",
                                    [(q, value) for q in qids])
                con.commit()
                settled += 1
                if settled % 20 == 0:
                    log(f"    resolved {settled}/{len(batches)} batches")
        stage_label_apply(con, log)
    # Record which quivers were still unknown when this stage ran. An export
    # made before the resolve stage existed shipped exactly these as
    # mutation_finite NULL, and a supplemental patch needs to know which rows
    # to correct without re-parsing gigabytes of part files.
    _mark(con, "resolve", {"settled": [q for q, _u in pending]})
    counts = con.execute("SELECT mutation_finite, count(*) FROM quivers GROUP BY 1").fetchall()
    log(f"  resolve: done {dict((k if k is not None else 'unknown', v) for k, v in counts)}")


def finite_class_seeds(con, n: int, log) -> list:
    """
    One seed per mutation-finite class: every quiver proved finite, deduped by
    exploring each class once.

    Mutation-finite quivers are vanishingly rare (428 of 42.5 M at rank 6), so
    this is cheap — and it is the only way to be sure the finite classes are in
    the dataset. Sampling cannot be trusted to find them: a class of 49 quivers
    is a 1-in-a-million target.
    """
    # Unordered for the same reason as stage_resolve: ORDER BY on the TEXT
    # primary key turns a sequential scan into one random seek per row.
    rows = sorted(con.execute(
        "SELECT id, upper FROM quivers WHERE mutation_finite = 1").fetchall())
    seen: set[str] = set()
    seeds = []
    for qid, upper in rows:
        if qid in seen:
            continue
        m = decode_upper(n, upper)
        # Proved finite already, so an uncapped walk terminates.
        seen |= set(_bfs_unlabeled(m, EXPLORE_BOUND, None).qid_set)
        seeds.append(m)
    from_cell = len(seeds)

    # The constructed seeds. A cell taken at |b_ij| <= 1 need not contain any
    # member of a finite class whose every quiver carries a double arrow, and a
    # SAMPLED cell (rank 8) need not contain one even when the full cell would
    # — so the Dynkin, affine and surface types go in by construction rather
    # than by discovery.
    for m in _curated_seeds(n, log):
        if quiver_id(m) in seen:
            continue
        orbit = _bfs_unlabeled(m, EXPLORE_BOUND, RESOLVE_CAP)
        if orbit.crossed or orbit.is_open:
            # Never explore an unproven seed uncapped below.
            log(f"    WARNING: curated seed {quiver_id(m)} did not drain under "
                f"cap {RESOLVE_CAP} (crossed={orbit.crossed}); skipped")
            continue
        seen |= set(orbit.qid_set)
        seeds.append(m)
    log(f"    {len(seeds)} mutation-finite class(es): {from_cell} found in the cell, "
        f"{len(seeds) - from_cell} constructed")
    return seeds


def stage_finite_classes(con, n: int, workers: int, la_timeout, known, log) -> dict:
    """
    Complete class rows for every mutation-finite class in the cell.

    These are the mathematically interesting classes (finite type, affine,
    surface and exceptional), and they are exactly the ones a uniform sample
    misses. Explored uncapped — finiteness is already proved, so the walk
    terminates — which makes every one of them `exploration = 'complete'` with
    its labeled orbit stored where it fits under LABELED_MAX.
    """
    seeds = finite_class_seeds(con, n, log)
    if not seeds:
        return {"mutation_classes": [], "quivers": [], "classes": {},
                "acyclicity_by_qid": {}, "rank_stats": {"labeled_quiver_count": 0}}

    def prog(stage, i, tot):
        if i == tot or i % max(1, tot // 5) == 0:
            log(f"    finite {stage}: {i}/{tot}")

    result = run_generation(max_vertices=n, bound=EXPLORE_BOUND, ranks=[n], node_cap=None,
                            seeds=seeds, workers=workers, progress=prog)
    rows = build_rank_rows(result, n, known_acyclicity=known, bound=2, node_cap=None,
                           generator="finite", census_size=None, la_timeout=la_timeout,
                           workers=workers, progress=prog)
    # Every member of a complete class needs a quiver row, including members
    # outside the cell — storing the FULL mutation class of every Dynkin and
    # surface type is the point. stage_export reads quiver rows straight from
    # the scratch table, so the strays are inserted here with the invariants
    # build_rank_rows already computed for them.
    added = 0
    for q in rows["quivers"]:
        present = con.execute("SELECT 1 FROM quivers WHERE id = ?", (q["id"],)).fetchone()
        if present is None:
            con.execute(
                "INSERT INTO quivers (id, upper, max_edge, is_acyclic, is_connected, "
                "is_bipartite, is_abundant, is_planar, representation_type, symmetry_group, "
                "mutation_finite, mutation_class_id, labeling_count, invariants_done, "
                "label_done) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,1)",
                (q["id"], q["exchange_matrix"], q["max_edge"], int(q["is_acyclic"]),
                 int(q["is_connected"]), _tri(q["is_bipartite"]), _tri(q["is_abundant"]),
                 _tri(q["is_planar"]), q["representation_type"],
                 json.dumps(q["symmetry_group"]) if q["symmetry_group"] else None,
                 _tri(q["mutation_finite"]), q["mutation_class_id"], q["labeling_count"]))
            added += 1
        else:
            con.execute("UPDATE quivers SET mutation_class_id=?, labeling_count=?, "
                        "mutation_finite=coalesce(mutation_finite, ?) WHERE id=?",
                        (q["mutation_class_id"], q["labeling_count"],
                         q["mutation_finite"], q["id"]))
    con.commit()
    log(f"    {len(rows['mutation_classes'])} finite class row(s), "
        f"{added} member(s) inserted from outside the cell")
    return rows


def _merge_class_rows(a: dict, b: dict) -> dict:
    """
    Union two build_rank_rows outputs; `b` wins an id collision.

    Only the class-side keys are merged. Quiver rows are not: stage_export
    reads those straight from the scratch table, which both stages have
    already written their membership back to.
    """
    by_id = {r["id"]: r for r in a["mutation_classes"]}
    by_id.update({r["id"]: r for r in b["mutation_classes"]})
    classes = {**a["classes"], **b["classes"]}
    return {
        "mutation_classes": sorted(by_id.values(), key=lambda r: r["id"]),
        "quivers": a["quivers"],
        "classes": classes,
        "acyclicity_by_qid": {**a["acyclicity_by_qid"], **b["acyclicity_by_qid"]},
        "rank_stats": {"labeled_quiver_count":
                       sum((c.labeled_size or 0) for c in classes.values())},
    }


# ---------------------------------------------------------------------------
# 4. sample: class rows for K quivers via the normal pipeline
# ---------------------------------------------------------------------------

def stage_sample(con, n: int, k: int, node_cap: int, workers: int, seed: int, la_timeout, known, log) -> dict:
    """
    Class rows for a uniform sample of k quivers, via the normal pipeline.

    **k is a memory budget, not a coverage knob.** run_generation holds every
    explored quiver in memory, and one rank-7 matrix costs 768 bytes:

        rank 6, 250k seeds  ->  2,395,384 quivers  ->  1.8 GB   (fine)
        rank 7, 250k seeds  -> 14,678,008 quivers  -> 11.3 GB   (thrashed 16 GB)

    Orbits at a cell bound below EXPLORE_BOUND escape the cell, so the same k
    explores ~6x more at rank 7 than at rank 6. Size k from the measured
    quivers-per-seed of the rank, not from what a previous rank used.

    Chunking k is NOT a way around this: the mc_id of a partially explored
    class is the lex-min over the members that were explored, so seeds of one
    class split across chunks mint two different ids instead of gluing into
    one. The sample has to fit in one run_generation call.
    """
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
    log(f"    {len(result.quivers)} quivers in {len(result.classes)} classes "
        f"(~{len(result.quivers) * 768 / 1e9:.1f} GB resident); class invariants ...")
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
    bad = con.execute("SELECT count(*) FROM quivers WHERE is_connected = 0").fetchone()[0]
    if bad:
        raise RuntimeError(f"{bad} disconnected quivers in the scratch table; the census is connected-only")
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
                 sample_k: int, part_bytes: int, log,
                 cell_sample: Optional[int] = None) -> list[dict]:
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
        "generator": (f"cell-sample:{cell_sample}" if cell_sample else "orderly")
                     + f"; classes for sample:{sample_k}",
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
                    resolve_cap: int = RESOLVE_CAP, cell_sample: Optional[int] = None,
                    sample: int = 1_000_000, sample_seed: int = 0, workers: int = 8,
                    la_timeout: Optional[float] = 1.0, part_bytes: int = DEFAULT_PART_BYTES,
                    log=_log) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"work-n{n}.sqlite")
    con = _db(path)
    log(f"  rank {n} (cell |b_ij| <= {h}): streaming pipeline in {path}")
    if cell_sample:
        stage_generate_sample(con, n, h, cell_sample, sample_seed, log)
    else:
        stage_generate(con, n, h, workers, log)
    stage_invariants(con, path, n, workers, log)
    stage_label(con, path, n, label_cap, workers, log)
    stage_resolve(con, n, resolve_cap, workers, log)

    known: dict = {}
    for j in range(1, n):
        ck = _load_json(os.path.join(out_dir, f"acyclicity-n{j}.json"))
        if ck is None:
            raise SystemExit(f"rank {n} needs acyclicity-n{j}.json — export the lower ranks first")
        known.update(ck)
    class_rows = stage_sample(con, n, sample, node_cap, workers, sample_seed, la_timeout, known, log)
    # Sampling cannot be relied on to find the mutation-finite classes, so they
    # are explored explicitly and merged in.
    class_rows = _merge_class_rows(
        class_rows, stage_finite_classes(con, n, workers, la_timeout, known, log))
    parts = stage_export(con, out_dir, n, h, class_rows, node_cap, sample, part_bytes, log,
                         cell_sample=cell_sample)
    _atomic_write(os.path.join(out_dir, f"acyclicity-n{n}.json"),
                  json.dumps(class_rows["acyclicity_by_qid"], sort_keys=True))

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = _load_json(manifest_path) or {"ranks": {}}
    manifest.setdefault("ranks", {})
    # pipeline_version is a TOP-LEVEL key (d1_export writes it there). Putting it
    # inside "ranks" makes the rank map contain a non-numeric key, which breaks
    # every consumer that iterates it — verify-export.py and import-d1.sh.
    manifest["pipeline_version"] = PIPELINE_VERSION
    manifest["ranks"][str(n)] = {
        "parts": parts,
        "depends_on": {f"acyclicity-n{j}.json": _sha256_file(os.path.join(out_dir, f"acyclicity-n{j}.json")) for j in range(1, n)},
        "settings": {"bound": h, "node_cap": node_cap, "label_cap": label_cap,
                     "resolve_cap": resolve_cap, "cell_sample": cell_sample,
                     "generator": "bigcell",
                     "sample": sample, "la_timeout": la_timeout, "schema": 3},
        "quiver_count": con.execute("SELECT count(*) FROM quivers").fetchone()[0],
        "class_count": len(class_rows["mutation_classes"]),
        # The three fields below are what the other ranks carry; without them the
        # rank-6 entry is not interchangeable with an export_ranks one.
        "labeled_quiver_count": sum(r["class_size"] or 0 for r in class_rows["mutation_classes"]),
        "truncated_classes": sum(1 for r in class_rows["mutation_classes"]
                                 if r["exploration"] == "truncated"),
        "census_size": census.count_connected_quivers(n, h),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    log(f"  rank {n}: wrote {len(parts)} part(s), {sum(p['bytes'] for p in parts) / 1e9:.2f} GB")
