"""
tests/test_export.py — qmd/d1_export.py (schema v2, multipart streaming SQL)

The SQL files are the only path from the math pipeline into production, so:
  * literals must escape correctly,
  * the column lists must match the Drizzle schema (src/db/schema.ts),
  * a rank must be idempotent (import twice == import once) and replace, not append,
  * no statement may exceed D1's per-statement limit, parts must respect the byte cap,
  * the resume logic must invalidate a rank when a lower-rank checkpoint changes,
  * a truncated class must never be stored as finite.
"""
import json
import os
import re
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmd import d1_export  # noqa: E402
from qmd.core import run_generation  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCHEMA_TS = os.path.join(ROOT, "src", "db", "schema.ts")
MIGRATIONS = os.path.join(ROOT, "drizzle")

D1_STATEMENT_LIMIT = 100_000     # bytes; D1 rejects larger single statements


def test_lit_escaping():
    lit = d1_export._lit
    assert lit(None) == "NULL"
    assert lit(True) == "1" and lit(False) == "0"
    assert lit(7) == "7" and lit(-3) == "-3"
    assert lit("it's") == "'it''s'"
    assert lit([[0, 1], [-1, 0]]) == "'[[0,1],[-1,0]]'"
    assert lit({"b": 1, "a": "x'y"}) == """'{"b":1,"a":"x''y"}'"""
    assert lit("é") == "'é'"


def _schema_columns():
    """table -> [column names] parsed from the Drizzle schema."""
    src = open(SCHEMA_TS, encoding="utf-8").read()
    tables = {}
    for m in re.finditer(r'sqliteTable\(\s*"(\w+)"', src):
        start = m.end()
        nxt = src.find("sqliteTable(", start)
        body = src[start: nxt if nxt != -1 else len(src)]
        tables[m.group(1)] = re.findall(r'(?:text|integer|real|blob)\(\s*"(\w+)"', body)
    return tables


def test_export_columns_match_drizzle_schema():
    cols = _schema_columns()
    assert set(d1_export._MC_COLUMNS) == set(cols["mutation_classes"])
    assert set(d1_export._LABELING_COLUMNS) == set(cols["labelings"])
    assert set(d1_export._FRONTIER_COLUMNS) == set(cols["frontier_quivers"])
    assert set(d1_export._QUIVER_COLUMNS) == set(cols["quivers"])
    assert set(d1_export._STATS_COLUMNS) == set(cols["rank_stats"])
    assert "mutation_class_payloads" not in cols


@pytest.fixture(scope="module")
def rank_rows():
    """Rank 1..3 rows exactly as export_ranks would build them."""
    out = {}
    known = {}
    for n in (1, 2, 3):
        r = run_generation(max_vertices=n, bound=2, ranks=[n])
        rows = d1_export.build_rank_rows(r, n, known_acyclicity=known, bound=2)
        known.update(rows["acyclicity_by_qid"])
        out[n] = rows
    return out


@pytest.fixture(scope="module")
def rank_sql(rank_rows):
    return {n: d1_export.render_rank_sql(n, rows, bound=2) for n, rows in rank_rows.items()}


def _fresh_db():
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    for name in sorted(os.listdir(MIGRATIONS)):
        if name.endswith(".sql"):
            migration = open(os.path.join(MIGRATIONS, name), encoding="utf-8").read()
            con.executescript(migration.replace("--> statement-breakpoint", ""))
    return con


def _snapshot(con):
    out = {}
    for t in ("mutation_classes", "labelings", "frontier_quivers", "quivers", "rank_stats"):
        out[t] = con.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall()
    return out


def test_rank_sql_is_idempotent_and_consistent(rank_sql):
    con = _fresh_db()
    for n in (1, 2, 3):
        con.executescript(rank_sql[n])
    once = _snapshot(con)
    con.executescript(rank_sql[3])          # re-import rank 3 only
    assert _snapshot(con) == once
    # rank_stats agree with the row tables
    for n in (1, 2, 3):
        qc, lc, cc = con.execute(
            "SELECT quiver_count, labeled_quiver_count, class_count FROM rank_stats WHERE n=?", (n,)).fetchone()
        assert qc == con.execute("SELECT count(*) FROM quivers WHERE n=?", (n,)).fetchone()[0]
        assert cc == con.execute("SELECT count(*) FROM mutation_classes WHERE n=?", (n,)).fetchone()[0]
        assert lc == con.execute(
            "SELECT count(*) FROM labelings l JOIN mutation_classes m ON m.id=l.mutation_class_id WHERE m.n=?",
            (n,)).fetchone()[0]
        assert lc == con.execute("SELECT sum(labeling_count) FROM quivers WHERE n=?", (n,)).fetchone()[0]
    assert con.execute("SELECT quiver_count FROM rank_stats WHERE n = 3").fetchone()[0] == 25
    # every quiver has a class, every labeling's quiver exists, ords are dense
    assert con.execute("SELECT count(*) FROM quivers WHERE mutation_class_id IS NULL").fetchone()[0] == 0
    assert con.execute(
        "SELECT count(*) FROM labelings l LEFT JOIN quivers q ON q.id=l.qmd_id WHERE q.id IS NULL").fetchone()[0] == 0
    bad = con.execute(
        "SELECT mutation_class_id FROM labelings GROUP BY mutation_class_id "
        "HAVING max(ord) != count(*) - 1 OR min(ord) != 0").fetchall()
    assert bad == []
    # labeling_offset is the prefix sum in id order within a rank
    rows = con.execute("SELECT n, labeling_count, labeling_offset FROM quivers ORDER BY n, id").fetchall()
    running = {}
    for n, cnt, off in rows:
        assert off == running.get(n, 0)
        running[n] = off + cnt
    # class-size per class equals its labelings
    mism = con.execute(
        "SELECT m.id FROM mutation_classes m JOIN (SELECT mutation_class_id, count(*) c FROM labelings "
        "GROUP BY mutation_class_id) l ON l.mutation_class_id=m.id WHERE l.c != m.class_size").fetchall()
    assert mism == []


def test_rank_sql_replaces_not_appends(rank_sql):
    con = _fresh_db()
    con.executescript(rank_sql[1]); con.executescript(rank_sql[2])
    con.execute("INSERT INTO mutation_classes (id, n, canonical_matrix, canonical_quiver_id, is_open, "
                "class_size, distinct_quiver_count, merged_orbit_count, is_infinite_expected) "
                "VALUES ('MC.n2.stale', 2, '[]', 'Q.n2.stale', 0, 1, 1, 1, 0)")
    con.execute("INSERT INTO labelings VALUES ('MC.n2.stale', 0, 'Q.n2.stale', '[]')")
    con.commit()
    con.executescript(rank_sql[2])
    assert con.execute("SELECT count(*) FROM mutation_classes WHERE id = 'MC.n2.stale'").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM labelings WHERE mutation_class_id = 'MC.n2.stale'").fetchone()[0] == 0


def test_statements_under_d1_limit(rank_sql):
    for sql in rank_sql.values():
        for stmt in sql.split(";\n"):
            assert len(stmt.encode("utf-8")) < D1_STATEMENT_LIMIT


def test_statement_byte_chunking():
    rows = [{"a": i, "b": "x" * 1000} for i in range(500)]
    stmts = list(d1_export._insert_stmts("t", ["a", "b"], rows, stmt_bytes=20_000))
    assert len(stmts) > 1
    assert all(len(s.encode()) <= 20_000 + 2 for s in stmts)
    assert sum(s.count("'x") for s in stmts) == 500
    with pytest.raises(ValueError):
        list(d1_export._insert_stmts("t", ["a"], [{"a": "y" * 200_000}]))


def test_multipart_writer_and_manifest(tmp_path, rank_rows):
    parts = d1_export.write_rank_sql(str(tmp_path), 3, rank_rows[3], bound=2, part_bytes=4000)
    assert len(parts) > 1
    for p in parts:
        data = open(tmp_path / p["file"], "rb").read()
        assert len(data) == p["bytes"] and p["bytes"] <= 4000 + D1_STATEMENT_LIMIT
        assert d1_export._sha256_file(str(tmp_path / p["file"])) == p["sha256"]
    # part 001 carries the deletes; the concatenation equals the single-string render
    first = open(tmp_path / parts[0]["file"], encoding="utf-8").read()
    assert "DELETE FROM mutation_classes WHERE n = 3;" in first
    joined = "".join(open(tmp_path / p["file"], encoding="utf-8").read() for p in parts)
    assert joined == d1_export.render_rank_sql(3, rank_rows[3], bound=2)


def test_export_ranks_resume_invalidates_dependents(tmp_path):
    logs = []
    d1_export.export_ranks(str(tmp_path), max_vertices=2, bound=2, log=logs.append)
    m1 = json.load(open(tmp_path / "manifest.json"))
    assert set(m1["ranks"]) == {"1", "2"}
    assert m1["ranks"]["2"]["depends_on"].keys() == {"acyclicity-n1.json"}
    # Nothing changed -> both skipped.
    logs.clear()
    d1_export.export_ranks(str(tmp_path), max_vertices=2, bound=2, log=logs.append)
    assert sum("skipping" in l for l in logs) == 2
    # Touch rank 1's checkpoint -> rank 2 must be regenerated.
    ck = tmp_path / "acyclicity-n1.json"
    ck.write_text(ck.read_text() + " ")
    logs.clear()
    d1_export.export_ranks(str(tmp_path), max_vertices=2, bound=2, ranks=[2], log=logs.append)
    assert any("regenerating" in l for l in logs) and not any("skipping" in l for l in logs)


def test_truncated_class_is_never_finite():
    r = run_generation(max_vertices=3, bound=2, ranks=[3], node_cap=3)
    rows = d1_export.build_rank_rows(r, 3, known_acyclicity={}, bound=2, node_cap=3)
    truncated = [c for c in rows["mutation_classes"] if c["exploration"] == "truncated"]
    assert truncated, "node cap 3 must truncate some rank-3 class"
    for c in truncated:
        assert c["is_open"] is True
        assert c["is_finite_confirmed"] is None and c["is_infinite_confirmed"] is None
        assert c["dynkin_type"] is None
        assert c["is_mutation_acyclic"] is not False          # never a proof of absence
        assert c["provenance"]["exploration"]["state"] == "truncated"
    assert rows["rank_stats"]["node_cap"] == 3


def test_exported_dist_files_under_d1_limit():
    """If a real export exists (dist/d1), every statement in it must fit too."""
    d = os.path.join(ROOT, "dist", "d1")
    files = [f for f in os.listdir(d) if f.endswith(".sql")] if os.path.isdir(d) else []
    if not files:
        pytest.skip("no dist/d1 export present")
    for f in files:
        sql = open(os.path.join(d, f), encoding="utf-8").read()
        worst = max(len(s.encode("utf-8")) for s in sql.split(";\n"))
        assert worst < D1_STATEMENT_LIMIT, f"{f}: largest statement {worst} bytes"
