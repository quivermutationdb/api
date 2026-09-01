"""
tests/test_census.py — qmd/census.py (exact counts, orderly generation, sampling)
and the parallel pipeline path.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmd import census  # noqa: E402
from qmd.core import canonical_form, generate_seed_quivers, is_connected, run_generation, to_matrix  # noqa: E402


@pytest.mark.parametrize("n,h,expected", [
    (1, 2, 1), (2, 2, 3), (3, 2, 25), (4, 2, 695),      # the published n<=4 census
    (3, 1, 7), (4, 1, 42), (5, 1, 582), (3, 10, 1561),
])
def test_count_quivers_known_values(n, h, expected):
    assert census.count_quivers(n, h) == expected


def test_count_matches_brute_force_small_cells():
    for n in (2, 3):
        for h in (1, 2, 3):
            seen = set()
            import itertools
            upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
            for combo in itertools.product(range(-h, h + 1), repeat=len(upper)):
                rows = [[0] * n for _ in range(n)]
                for (i, j), v in zip(upper, combo):
                    rows[i][j], rows[j][i] = v, -v
                seen.add(canonical_form(to_matrix(rows)))
            assert len(seen) == census.count_quivers(n, h), (n, h)


def test_orderly_generation_equals_brute_force():
    brute = set(generate_seed_quivers(4, 2))
    for n in range(1, 5):
        cell = set(census.census_seeds(n, 2, connected_only=False))
        assert cell == {m for m in brute if len(m) == n}
        assert len(cell) == census.count_quivers(n, 2)
        conn = set(census.census_seeds(n, 2))
        assert conn == {m for m in cell if is_connected(m)}
        assert len(conn) == census.count_connected_quivers(n, 2)


@pytest.mark.parametrize("n,h,expected", [(3, 2, 22), (4, 2, 667), (5, 2, 82141), (4, 1, 34)])
def test_count_connected_quivers(n, h, expected):
    assert census.count_connected_quivers(n, h) == expected


def test_count_connected_matches_brute_force():
    import itertools
    for n, h in [(3, 3), (4, 1)]:
        seen = set()
        upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for combo in itertools.product(range(-h, h + 1), repeat=len(upper)):
            rows = [[0] * n for _ in range(n)]
            for (i, j), v in zip(upper, combo):
                rows[i][j], rows[j][i] = v, -v
            m = to_matrix(rows)
            if is_connected(m):
                seen.add(canonical_form(m))
        assert len(seen) == census.count_connected_quivers(n, h), (n, h)


@pytest.mark.parametrize("n,h", [(4, 3), (5, 1), (3, 5)])
def test_orderly_generation_count_and_canonicity(n, h):
    reps = list(census.generate_cell(n, h))
    assert len(reps) == census.count_quivers(n, h)
    assert all(census.is_key_minimal(m) for m in reps)
    assert len({canonical_form(m) for m in reps}) == len(reps)   # distinct classes


def test_key_minimal_form_is_isomorphism_invariant():
    import random
    rng = random.Random(11)
    for _ in range(100):
        n = rng.randint(2, 5)
        m = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                v = rng.randint(-2, 2)
                m[i][j], m[j][i] = v, -v
        m = to_matrix(m)
        p = list(range(n)); rng.shuffle(p)
        m2 = tuple(tuple(m[p[i]][p[j]] for j in range(n)) for i in range(n))
        assert census.key_minimal_form(m) == census.key_minimal_form(m2)


def test_orderly_generation_parallel_matches_serial():
    serial = census.census_seeds(4, 2)
    parallel = census.census_seeds(4, 2, workers=3)
    assert serial == parallel


def test_sample_cell_is_distinct_canonical_and_in_cell():
    s = census.sample_cell(5, 3, 50, seed=1, connected_only=True)
    assert all(is_connected(m) for m in s)
    assert len(s) == 50 and len(set(s)) == 50
    assert all(canonical_form(m) == m for m in s)
    assert all(max(abs(x) for row in m for x in row) <= 3 for m in s)
    assert census.sample_cell(5, 3, 50, seed=1, connected_only=True) == s          # deterministic


def test_parallel_run_generation_matches_serial():
    seeds = census.census_seeds(4, 2, connected_only=False)
    a = run_generation(max_vertices=4, bound=2, ranks=[4], seeds=seeds)
    b = run_generation(max_vertices=4, bound=2, ranks=[4], seeds=seeds, workers=4)
    assert set(a.quivers) == set(b.quivers)
    assert set(a.classes) == set(b.classes)
    assert a.membership == b.membership
    for k in a.classes:
        assert a.classes[k].members == b.classes[k].members
        assert a.classes[k].labeled_quivers == b.classes[k].labeled_quivers
        assert a.classes[k].exploration == b.classes[k].exploration
    assert (a.open_open_gluings, a.closed_closed_merges) == (b.open_open_gluings, b.closed_closed_merges)


def test_bigcell_pipeline_matches_normal_pipeline(tmp_path):
    """The streaming (scratch-SQLite) pipeline on (4,2) must reproduce the
    normal export: same connected quivers, same finiteness labels, same ids."""
    import json, os, sqlite3
    from qmd import bigcell, d1_export
    logs = []
    # lower ranks via the normal path (checkpoints), then rank 4 via bigcell
    d1_export.export_ranks(str(tmp_path), max_vertices=3, bound=2, log=logs.append)
    bigcell.export_big_cell(str(tmp_path), n=4, h=2, label_cap=20, node_cap=100,
                            sample=667, workers=2, la_timeout=0, log=logs.append)
    ref_dir = tmp_path / "ref"
    d1_export.export_ranks(str(ref_dir), max_vertices=4, bound=2, log=logs.append)

    def load(dirpath, n):
        con = sqlite3.connect(":memory:")
        for name in sorted(os.listdir(os.path.join(os.path.dirname(__file__), "..", "drizzle"))):
            if name.endswith(".sql"):
                con.executescript(open(os.path.join(os.path.dirname(__file__), "..", "drizzle", name)).read().replace("--> statement-breakpoint", ""))
        m = json.load(open(os.path.join(dirpath, "manifest.json")))
        for part in m["ranks"][str(n)]["parts"]:
            con.executescript(open(os.path.join(dirpath, part["file"])).read())
        return con
    a = load(str(tmp_path), 4); b = load(str(ref_dir), 4)
    qa = a.execute("SELECT id, exchange_matrix, mutation_finite, max_edge, is_acyclic, representation_type FROM quivers ORDER BY id").fetchall()
    qb = b.execute("SELECT id, exchange_matrix, mutation_finite, max_edge, is_acyclic, representation_type FROM quivers ORDER BY id").fetchall()
    assert len(qa) == 667 and qa == qb
    ca = {r[0] for r in a.execute("SELECT id FROM mutation_classes")}
    cb = {r[0] for r in b.execute("SELECT id FROM mutation_classes")}
    assert ca == cb                                   # sample = whole cell -> identical classes
    assert a.execute("SELECT count(*) FROM labelings").fetchone()[0] == b.execute("SELECT count(*) FROM labelings").fetchone()[0]

    # The manifests must be interchangeable: bigcell claims "the same manifest
    # format as export_ranks", and import-d1.sh/verify-export.py both ITERATE
    # manifest["ranks"], so a stray non-numeric key there breaks the import.
    # Reading one rank by key (as the loader above does) hides exactly that bug.
    ma = json.load(open(os.path.join(str(tmp_path), "manifest.json")))
    mb = json.load(open(os.path.join(str(ref_dir), "manifest.json")))
    assert ma["pipeline_version"] == mb["pipeline_version"]     # top level, not inside "ranks"
    assert all(k.isdigit() for k in ma["ranks"]), sorted(ma["ranks"])
    assert sorted(ma["ranks"]["4"]) == sorted(mb["ranks"]["4"]), "rank entry shape differs"
    assert ma["ranks"]["4"]["census_size"] == mb["ranks"]["4"]["census_size"]


def test_curated_seeds_are_connected():
    import json
    doc = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "seeds.json")))
    for e in doc["seeds"]:
        assert is_connected(to_matrix(e["matrix"])), e["name"]


def test_exporter_refuses_disconnected_quivers():
    """A disconnected quiver can only reach the exporter through a bug; it must stop the export."""
    from qmd import d1_export
    r = run_generation(max_vertices=3, bound=2, ranks=[3], seeds=[to_matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])])
    with pytest.raises(RuntimeError, match="disconnected"):
        d1_export.build_rank_rows(r, 3, known_acyclicity={}, bound=2)


def test_bigcell_generate_stage_is_resumable(tmp_path):
    """
    A kill during `generate` must not discard the parents already extended:
    on restart the committed parents are skipped, the uncommitted ones are
    redone, and the cell still comes out exactly right.
    """
    from qmd import bigcell
    con = bigcell._db(str(tmp_path / "work.sqlite"))
    logs = []
    bigcell.stage_generate(con, 4, 2, 2, logs.append)
    expected = census.count_connected_quivers(4, 2)
    assert con.execute("SELECT count(*) FROM quivers").fetchone()[0] == expected
    all_parents = con.execute("SELECT count(*) FROM parents_done").fetchone()[0]
    assert all_parents == census.count_quivers(3, 2)          # every rank-3 parent recorded

    # Simulate a crash mid-stage: the stage marker and the last parents' progress
    # are gone, but the quivers they already committed are still there.
    con.execute("DELETE FROM stages")
    con.execute("DELETE FROM parents_done WHERE idx >= ?", (all_parents // 2,))
    con.commit()
    kept = con.execute("SELECT count(*) FROM parents_done").fetchone()[0]

    logs.clear()
    bigcell.stage_generate(con, 4, 2, 2, logs.append)
    assert any(f"{kept} already extended" in l for l in logs), logs[:3]
    assert con.execute("SELECT count(*) FROM quivers").fetchone()[0] == expected
    assert con.execute("SELECT count(*) FROM parents_done").fetchone()[0] == all_parents
    assert bigcell._stage_done(con, "generate")

    # A finished stage is a no-op, and never deletes what is stored.
    logs.clear()
    bigcell.stage_generate(con, 4, 2, 2, logs.append)
    assert logs == ["  generate: done"]
    assert con.execute("SELECT count(*) FROM quivers").fetchone()[0] == expected


def test_bigcell_label_stage_is_resumable_and_stages_verdicts(tmp_path):
    """
    The label stage must survive a kill without recomputing what it settled,
    and must never write verdicts into `quivers` row by row.

    Scattered `UPDATE quivers ... WHERE id = ?` against a multi-GB table costs
    one random page read per row and stalled the rank-6 run; verdicts are
    staged in `label_verdicts` and folded in one id-ordered pass instead.
    """
    from qmd import bigcell
    path = str(tmp_path / "work.sqlite")
    con = bigcell._db(path)
    logs = []
    bigcell.stage_generate(con, 4, 2, 2, logs.append)
    bigcell.stage_invariants(con, path, 4, 2, logs.append)

    bigcell.stage_label(con, path, 4, cap=20, workers=2, log=logs.append)
    full = dict(con.execute("SELECT id, mutation_finite FROM quivers").fetchall())
    assert bigcell._stage_done(con, "label")
    # Every verdict was drained out of the staging table.
    assert con.execute("SELECT count(*) FROM label_verdicts").fetchone()[0] == 0
    # Rank 4 at bound 2 is settled either way for every quiver.
    assert all(v is not None for v in full.values())

    # Simulate a kill: drop the stage marker and the watermark, and unsettle the
    # tail half of the table (as if those rows had never been reached).
    ids = sorted(full)
    tail = ids[len(ids) // 2:]
    con.execute("DELETE FROM stages WHERE name = 'label'")
    con.execute("DELETE FROM watermarks")
    con.executemany("UPDATE quivers SET mutation_finite = NULL, label_done = 0 WHERE id = ?",
                    [(q,) for q in tail])
    con.commit()

    logs.clear()
    bigcell.stage_label(con, path, 4, cap=20, workers=2, log=logs.append)
    assert dict(con.execute("SELECT id, mutation_finite FROM quivers").fetchall()) == full
    # The settled prefix was skipped via the watermark, not re-explored.
    assert any("skipped" in l and "settled" in l for l in logs), logs
    assert con.execute("SELECT count(*) FROM label_verdicts").fetchone()[0] == 0

    # A finished stage is a no-op.
    logs.clear()
    bigcell.stage_label(con, path, 4, cap=20, workers=2, log=logs.append)
    assert logs == ["  label: done"]


def test_bigcell_watermark_only_advances_over_settled_runs(tmp_path):
    """
    The watermark is the resume point, so it must never move past a quiver that
    still needs work — including one left unsettled in the middle of the prefix.
    """
    from qmd import bigcell
    path = str(tmp_path / "work.sqlite")
    con = bigcell._db(path)
    logs = []
    bigcell.stage_generate(con, 4, 2, 2, logs.append)
    bigcell.stage_invariants(con, path, 4, 2, logs.append)
    con.execute("UPDATE quivers SET mutation_finite = 0, label_done = 1")
    ids = [r[0] for r in con.execute("SELECT id FROM quivers ORDER BY id").fetchall()]
    hole = ids[len(ids) // 3]
    con.execute("UPDATE quivers SET mutation_finite = NULL, label_done = 0 WHERE id = ?", (hole,))
    con.commit()

    stop = bigcell._seed_label_watermark(con, path, logs.append)
    assert stop < hole, (stop, hole)
    assert bigcell._watermark(con, "label") == stop


def test_bigcell_resolves_unknowns_and_stores_every_finite_class(tmp_path):
    """
    A mutation-finite class larger than the label cap can never drain, so the
    capped label pass returns *unknown* for every one of its members — and a
    uniform sample is far too thin to rescue them. Rank 6 shipped A6/D6/E6 only
    because the sample happened to land in all three (a ~3% event) and shipped
    ten other finite classes not at all.

    Pin both halves of the fix: nothing is left unknown, and every finite class
    gets a complete row with its labelings.
    """
    import json, os, sqlite3
    from qmd import bigcell, d1_export, dynkin
    logs = []
    d1_export.export_ranks(str(tmp_path), max_vertices=3, bound=2, log=logs.append)
    # label_cap=2 is far too small for any finite rank-4 class to drain, and
    # sample=1 means the sample is essentially certain not to find one either.
    bigcell.export_big_cell(str(tmp_path), n=4, h=2, label_cap=2, node_cap=100,
                            sample=1, workers=2, la_timeout=0, log=logs.append)

    con = sqlite3.connect(":memory:")
    drizzle = os.path.join(os.path.dirname(__file__), "..", "drizzle")
    for name in sorted(os.listdir(drizzle)):
        if name.endswith(".sql"):
            con.executescript(open(os.path.join(drizzle, name)).read()
                              .replace("--> statement-breakpoint", ""))
    manifest = json.load(open(os.path.join(str(tmp_path), "manifest.json")))
    for part in manifest["ranks"]["4"]["parts"]:
        con.executescript(open(os.path.join(str(tmp_path), part["file"])).read())

    assert con.execute(
        "SELECT count(*) FROM quivers WHERE mutation_finite IS NULL").fetchone()[0] == 0

    stored = dict(con.execute("SELECT id, exploration FROM mutation_classes").fetchall())
    for mc_id, name in dynkin.reference_for(4).items():
        assert mc_id in stored, f"finite class {name} ({mc_id}) has no row"
        assert stored[mc_id] == "complete", f"{name} stored as {stored[mc_id]!r}"
        assert con.execute("SELECT count(*) FROM labelings WHERE mutation_class_id = ?",
                           (mc_id,)).fetchone()[0] > 0, f"{name} has no labelings"
        assert con.execute(
            "SELECT count(*) FROM quivers WHERE mutation_class_id = ? AND mutation_finite = 1",
            (mc_id,)).fetchone()[0] > 0, f"{name} members not flagged finite"
