"""
tests/test_export.py — qmd/d1_export.py (schema v3: compact matrices, sharded
multipart SQL, labelings only for complete classes, per-quiver finiteness).
"""
import json
import os
import re
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmd import d1_export  # noqa: E402
from qmd.core import is_connected, run_generation, to_matrix  # noqa: E402
from qmd.encoding import decode_upper, encode_upper  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SCHEMA_TS = os.path.join(ROOT, "src", "db", "schema.ts")
MIGRATIONS = os.path.join(ROOT, "drizzle")
D1_STATEMENT_LIMIT = 100_000


def test_lit_escaping():
    lit = d1_export._lit
    assert lit(None) == "NULL"
    assert lit(True) == "1" and lit(False) == "0"
    assert lit(7) == "7" and lit(-3) == "-3"
    assert lit("it's") == "'it''s'"
    assert lit({"b": 1, "a": "x'y"}) == """'{"b":1,"a":"x''y"}'"""


def test_upper_encoding_roundtrip():
    m = to_matrix([[0, 1, -2, 0], [-1, 0, 3, 1], [2, -3, 0, -1], [0, -1, 1, 0]])
    assert encode_upper(m) == "1,-2,0,3,1,-1"
    assert decode_upper(4, encode_upper(m)) == m
    assert decode_upper(1, "") == ((0,),)


def _schema_columns():
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
    assert set(d1_export._QUIVER_COLUMNS) == set(cols["quivers"])
    assert set(d1_export._STATS_COLUMNS) == set(cols["rank_stats"])
    assert "frontier_quivers" not in cols and "mutation_class_payloads" not in cols


def test_shard_routing_matches_config():
    assert d1_export.shard_of("Q.n4.7abc000000000000", 4) == ("main", "qmd")
    assert d1_export.shard_of("Q.n6.0abc000000000000", 6)[0] == "n6.0"
    assert d1_export.shard_of("Q.n6.1abc000000000000", 6)[0] == "n6.1"
    assert d1_export.shard_of("Q.n6.fabc000000000000", 6)[0] == "n6.3"   # 15 % 4
    assert d1_export.shard_of("Q.n6.7abc000000000000", 6)[0] == "n6.3"   # 7 % 4
    assert len(d1_export.shard_keys_for(6)) == 4


def _gen(n, **kw):
    """run_generation over the CONNECTED seeds of (n, 2), as the exporter does."""
    from qmd.census import census_seeds
    return run_generation(max_vertices=n, bound=2, ranks=[n], seeds=census_seeds(n, 2), **kw)


@pytest.fixture(scope="module")
def rank_rows():
    out, known = {}, {}
    for n in (1, 2, 3):
        r = _gen(n)
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
            con.executescript(open(os.path.join(MIGRATIONS, name), encoding="utf-8").read()
                              .replace("--> statement-breakpoint", ""))
    return con


def _load(con, sql_by_shard):
    for key in sorted(sql_by_shard):
        con.executescript(sql_by_shard[key])


def _snapshot(con):
    return {t: con.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall()
            for t in ("mutation_classes", "labelings", "quivers", "rank_stats")}


def test_rank_sql_is_idempotent_and_consistent(rank_sql):
    con = _fresh_db()
    for n in (1, 2, 3):
        _load(con, rank_sql[n])
    once = _snapshot(con)
    _load(con, rank_sql[3])
    assert _snapshot(con) == once
    for n in (1, 2, 3):
        qc, lc, cc = con.execute(
            "SELECT quiver_count, labeled_quiver_count, class_count FROM rank_stats WHERE n=?", (n,)).fetchone()
        assert qc == con.execute("SELECT count(*) FROM quivers WHERE n=?", (n,)).fetchone()[0]
        assert cc == con.execute("SELECT count(*) FROM mutation_classes WHERE n=?", (n,)).fetchone()[0]
        assert lc == con.execute("SELECT sum(labeling_count) FROM quivers WHERE n=? AND mutation_class_id IS NOT NULL", (n,)).fetchone()[0]
    assert con.execute("SELECT quiver_count FROM rank_stats WHERE n = 3").fetchone()[0] == 22   # connected rank-3 quivers
    # labelings exist exactly for complete classes, dense ords, matching class_size
    bad = con.execute("""
        SELECT m.id FROM mutation_classes m LEFT JOIN
          (SELECT mutation_class_id, count(*) c, max(ord) mx, min(ord) mn FROM labelings GROUP BY 1) l
          ON l.mutation_class_id = m.id
        WHERE (m.exploration = 'complete' AND (l.c IS NULL OR l.c != m.class_size OR l.mx != l.c - 1 OR l.mn != 0))
           OR (m.exploration != 'complete' AND l.c IS NOT NULL)""").fetchall()
    assert bad == []
    # every labeling's quiver exists; compact matrices decode to the stored rank
    assert con.execute("SELECT count(*) FROM labelings l LEFT JOIN quivers q ON q.id=l.qmd_id WHERE q.id IS NULL").fetchone()[0] == 0
    for n, enc in con.execute("SELECT n, exchange_matrix FROM quivers").fetchall():
        decode_upper(n, enc)
    # per-quiver finiteness agrees with the class trichotomy
    mism = con.execute("""
        SELECT q.id FROM quivers q JOIN mutation_classes m ON m.id = q.mutation_class_id
        WHERE (m.is_finite_confirmed = 1 AND q.mutation_finite IS NOT 1)
           OR (m.is_infinite_confirmed = 1 AND q.mutation_finite IS NOT 0)""").fetchall()
    assert mism == []
    # connected-only guarantee: every stored quiver and every stored labeling is connected
    for n, enc in con.execute("SELECT n, exchange_matrix FROM quivers").fetchall():
        assert is_connected(decode_upper(n, enc))
    assert con.execute("SELECT count(*) FROM quivers WHERE is_connected = 0").fetchone()[0] == 0
    # rows are inserted in id order, so (n, rowid) is id order (rowid-based keyset relies on it)
    ids = con.execute("SELECT id FROM quivers ORDER BY n, rowid").fetchall()
    assert ids == sorted(ids, key=lambda r: (int(r[0].split(".")[1][1:]), r[0]))


def test_rank_sql_replaces_not_appends(rank_sql):
    con = _fresh_db()
    _load(con, rank_sql[1]); _load(con, rank_sql[2])
    con.execute("INSERT INTO mutation_classes (id, n, canonical_matrix, canonical_quiver_id, is_open, "
                "class_size, distinct_quiver_count, merged_orbit_count, is_infinite_expected) "
                "VALUES ('MC.n2.stale', 2, '', 'Q.n2.stale', 0, 1, 1, 1, 0)")
    con.execute("INSERT INTO labelings VALUES ('MC.n2.stale', 0, 'Q.n2.stale', '')")
    con.commit()
    _load(con, rank_sql[2])
    assert con.execute("SELECT count(*) FROM mutation_classes WHERE id = 'MC.n2.stale'").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM labelings WHERE mutation_class_id = 'MC.n2.stale'").fetchone()[0] == 0


def test_statements_under_d1_limit(rank_sql):
    for by_shard in rank_sql.values():
        for sql in by_shard.values():
            for stmt in sql.split(";\n"):
                assert len(stmt.encode("utf-8")) < D1_STATEMENT_LIMIT


def test_statement_byte_chunking():
    rows = [{"a": i, "b": "x" * 1000} for i in range(500)]
    stmts = list(d1_export._insert_stmts("t", ["a", "b"], rows, stmt_bytes=20_000))
    assert len(stmts) > 1 and all(len(s.encode()) <= 20_002 for s in stmts)
    assert sum(s.count("'x") for s in stmts) == 500
    with pytest.raises(ValueError):
        list(d1_export._insert_stmts("t", ["a"], [{"a": "y" * 200_000}]))


def test_multipart_writer_and_manifest(tmp_path, rank_rows):
    parts = d1_export.write_rank_sql(str(tmp_path), 3, rank_rows[3], bound=2, part_bytes=4000)
    data_parts = [p for p in parts if ".main." in p["file"]]
    assert len(data_parts) > 1 and all(p["shard"] == "main" and p["database"] == "qmd" for p in parts)
    for p in parts:
        raw = open(tmp_path / p["file"], "rb").read()
        assert len(raw) == p["bytes"] and p["bytes"] <= 4000 + D1_STATEMENT_LIMIT
        assert d1_export._sha256_file(str(tmp_path / p["file"])) == p["sha256"]
    first = open(tmp_path / data_parts[0]["file"], encoding="utf-8").read()
    assert "DELETE FROM mutation_classes WHERE n = 3;" in first
    joined = "".join(open(tmp_path / p["file"], encoding="utf-8").read() for p in data_parts)
    assert joined == d1_export.render_rank_sql(3, rank_rows[3], bound=2)["main"]
    assert any(p["file"].endswith(".stats.001.sql") for p in parts)


def test_split_rank_parts_go_to_their_shards(rank_rows):
    """Render rank-3 rows as if rank 3 were split: every row lands in exactly one shard."""
    import qmd.d1_export as d
    orig = d._shards_config
    d._shards_config = lambda: {"main": {"binding": "DB", "database": "qmd"},
                                "split": {"3": {"buckets": 2, "databases": [
                                    {"binding": "A", "database": "qmd-a"}, {"binding": "B", "database": "qmd-b"}]}}}
    try:
        by_shard = d.render_rank_sql(3, rank_rows[3], bound=2)
        assert set(by_shard) == {"n3.0", "n3.1", "stats"}
        con_a, con_b = _fresh_db(), _fresh_db()
        con_a.executescript(by_shard["n3.0"]); con_b.executescript(by_shard["n3.1"])
        qa = {r[0] for r in con_a.execute("SELECT id FROM quivers")}
        qb = {r[0] for r in con_b.execute("SELECT id FROM quivers")}
        assert qa.isdisjoint(qb) and len(qa | qb) == 22
        assert all(int(i.split(".")[2][0], 16) % 2 == 0 for i in qa)
        # classes and their labelings travel together
        for con in (con_a, con_b):
            assert con.execute("SELECT count(*) FROM labelings l LEFT JOIN mutation_classes m ON m.id=l.mutation_class_id WHERE m.id IS NULL").fetchone()[0] == 0
    finally:
        d._shards_config = orig


def test_export_ranks_resume_invalidates_dependents(tmp_path):
    logs = []
    d1_export.export_ranks(str(tmp_path), max_vertices=2, bound=2, log=logs.append)
    m1 = json.load(open(tmp_path / "manifest.json"))
    assert set(m1["ranks"]) == {"1", "2"}
    logs.clear()
    d1_export.export_ranks(str(tmp_path), max_vertices=2, bound=2, log=logs.append)
    assert sum("skipping" in l for l in logs) == 2
    ck = tmp_path / "acyclicity-n1.json"
    ck.write_text(ck.read_text() + " ")
    logs.clear()
    d1_export.export_ranks(str(tmp_path), max_vertices=2, bound=2, ranks=[2], log=logs.append)
    assert any("regenerating" in l for l in logs) and not any("skipping" in l for l in logs)


def test_truncated_class_is_never_finite():
    r = _gen(3, node_cap=3)
    rows = d1_export.build_rank_rows(r, 3, known_acyclicity={}, bound=2, node_cap=3)
    truncated = [c for c in rows["mutation_classes"] if c["exploration"] == "truncated"]
    assert truncated, "node cap 3 must truncate some rank-3 class"
    for c in truncated:
        assert c["is_open"] is True
        assert c["is_finite_confirmed"] is None and c["is_infinite_confirmed"] is None
        assert c["dynkin_type"] is None and c["is_mutation_acyclic"] is not False
    qrows = {q["id"]: q for q in rows["quivers"]}
    for c in truncated:
        assert qrows[c["canonical_quiver_id"]]["mutation_finite"] is None


def test_derksen_owen_shortcut_marks_infinite_without_exploring(tmp_path):
    """A (3, 3) cell: seeds with an entry 3 become quiver-only rows with mutation_finite = 0."""
    logs = []
    d1_export.export_ranks(str(tmp_path), max_vertices=3, bound=3, log=logs.append)
    con = _fresh_db()
    m = json.load(open(tmp_path / "manifest.json"))
    for part in m["ranks"]["3"]["parts"]:
        con.executescript(open(tmp_path / part["file"], encoding="utf-8").read())
    from qmd.census import count_connected_quivers
    total = con.execute("SELECT count(*) FROM quivers WHERE n=3").fetchone()[0]
    assert total == count_connected_quivers(3, 3)
    do = con.execute("SELECT count(*) FROM quivers WHERE n=3 AND max_edge >= 3").fetchone()[0]
    assert do == total - count_connected_quivers(3, 2)
    assert con.execute("SELECT count(*) FROM quivers WHERE n=3 AND max_edge >= 3 AND mutation_finite = 0 AND mutation_class_id IS NULL").fetchone()[0] == do
    assert con.execute("SELECT count(*) FROM quivers WHERE is_connected = 0").fetchone()[0] == 0
    assert any("Derksen" in l for l in logs)


def test_exported_dist_files_under_d1_limit():
    d = os.path.join(ROOT, "dist", "d1")
    files = [f for f in os.listdir(d) if f.endswith(".sql")] if os.path.isdir(d) else []
    if not files:
        pytest.skip("no dist/d1 export present")
    for f in files:
        sql = open(os.path.join(d, f), encoding="utf-8").read()
        worst = max(len(s.encode("utf-8")) for s in sql.split(";\n"))
        assert worst < D1_STATEMENT_LIMIT, f"{f}: largest statement {worst} bytes"
