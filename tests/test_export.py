"""
tests/test_export.py — qmd/d1_export.py

The SQL files are the only path from the math pipeline into production, so:
  * literals must escape correctly,
  * the column lists must match the Drizzle schema (src/db/schema.ts),
  * each rank file must be idempotent (import twice == import once),
  * no single statement may exceed D1's per-statement size limit.
"""
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
MIGRATION = os.path.join(ROOT, "drizzle", "0000_init_schema.sql")

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
    assert set(d1_export._PAYLOAD_COLUMNS) == set(cols["mutation_class_payloads"])
    assert set(d1_export._QUIVER_COLUMNS) == set(cols["quivers"])
    assert set(d1_export._STATS_COLUMNS) == set(cols["rank_stats"])


@pytest.fixture(scope="module")
def rank_sql():
    """Rank-1 and rank-2 SQL exactly as export_ranks would produce them."""
    r1 = run_generation(max_vertices=1, bound=2, ranks=[1])
    rows1 = d1_export.build_rank_rows(r1, 1)
    r2 = run_generation(max_vertices=2, bound=2, ranks=[2])
    rows2 = d1_export.build_rank_rows(r2, 2, known_acyclicity=rows1["acyclicity_by_qid"])
    return {1: d1_export.render_rank_sql(1, rows1, bound=2),
            2: d1_export.render_rank_sql(2, rows2, bound=2)}


def _fresh_db():
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    migration = open(MIGRATION, encoding="utf-8").read().replace("--> statement-breakpoint", "")
    con.executescript(migration)
    return con


def _snapshot(con):
    out = {}
    for t in ("mutation_classes", "mutation_class_payloads", "quivers", "rank_stats"):
        out[t] = con.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall()
    return out


def test_rank_sql_is_idempotent(rank_sql):
    con = _fresh_db()
    con.executescript(rank_sql[1])
    con.executescript(rank_sql[2])
    once = _snapshot(con)
    con.executescript(rank_sql[2])          # re-import rank 2 only
    twice = _snapshot(con)
    assert once == twice
    assert con.execute("SELECT quiver_count FROM rank_stats WHERE n = 2").fetchone()[0] == 3
    assert con.execute("SELECT count(*) FROM quivers WHERE n = 2").fetchone()[0] == 3
    # every quiver's class exists (FK) and every payload has its class
    assert con.execute("SELECT count(*) FROM quivers WHERE mutation_class_id IS NULL").fetchone()[0] == 0


def test_rank_sql_replaces_not_appends(rank_sql):
    con = _fresh_db()
    con.executescript(rank_sql[1])
    con.executescript(rank_sql[2])
    # Plant a stale rank-2 row; a re-import must remove it.
    con.execute("INSERT INTO mutation_classes (id, n, canonical_matrix, canonical_quiver_id, "
                "is_open, class_size, distinct_quiver_count, merged_orbit_count, "
                "is_infinite_expected) VALUES ('MC.n2.stale', 2, '[]', 'Q.n2.stale', 0, 1, 1, 1, 0)")
    con.commit()
    con.executescript(rank_sql[2])
    assert con.execute("SELECT count(*) FROM mutation_classes WHERE id = 'MC.n2.stale'").fetchone()[0] == 0


def test_statements_under_d1_limit(rank_sql):
    for sql in rank_sql.values():
        for stmt in sql.split(";\n"):
            assert len(stmt.encode("utf-8")) < D1_STATEMENT_LIMIT


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
