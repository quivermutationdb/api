#!/usr/bin/env python3
"""
Render a QMD dataset as PostgreSQL COPY text — one .tsv per table, plus a
.copy file holding the exact COPY commands.

Reads a SQLite database carrying the QMD schema (the local D1, or a plain
SQLite fed the dist/d1 parts) rather than parsing the exported INSERT
statements: SQLite is the reader, so the values are the ones D1 would have
stored, not a re-parse of SQL text.

Column ORDER comes from qmd.d1_export, the same lists that render the D1
INSERTs, so the two exports cannot drift.

COPY text format (the default): tab-separated, \\N for NULL, and backslash /
tab / newline / carriage-return escaped. Booleans need no translation --
Postgres accepts '1' and '0' as boolean input, which is what SQLite stores.
JSON columns are already JSON text in SQLite and are read straight into jsonb.

`seq` is deliberately absent: it is assigned after the load with
row_number() OVER (PARTITION BY n ORDER BY id).

    python scripts/pg-export.py <out-dir> <sqlite-db>... [--gzip]

A sharded rank MUST pass every shard in one invocation, or seq restarts per
shard and rank_stats collides on its primary key.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmd.d1_export import (  # noqa: E402
    _LABELING_COLUMNS, _MC_COLUMNS, _QUIVER_COLUMNS, _STATS_COLUMNS,
)

# class_nicknames is rendered by scripts/nicknames.py, not d1_export, so its
# column list lives here. Keep in step with src/db/schema.ts.
_NICK_COLUMNS = ["mc_id", "nickname", "slug", "note", "added_by", "added_at"]

TABLES: dict[str, list[str]] = {
    "mutation_classes": _MC_COLUMNS,
    "quivers": _QUIVER_COLUMNS,
    "labelings": _LABELING_COLUMNS,
    "rank_stats": _STATS_COLUMNS,
    "class_nicknames": _NICK_COLUMNS,
}

# Load order: labelings references mutation_classes, so classes go first.
ORDER = ["rank_stats", "mutation_classes", "quivers", "labelings", "class_nicknames"]


def pg_text(v) -> str:
    r"""One COPY field. Backslash MUST be escaped before the others."""
    if v is None:
        return r"\N"
    if isinstance(v, bytes):
        v = v.decode("utf-8")
    s = str(v)
    if isinstance(v, bool):                    # sqlite3 never yields bool, but be explicit
        return "t" if v else "f"
    return (s.replace("\\", "\\\\")
             .replace("\t", "\\t")
             .replace("\n", "\\n")
             .replace("\r", "\\r"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("db", nargs="+",
                    help="one or more SQLite sources; a SHARDED rank must pass all of its "
                         "shards in one invocation (see below)")
    ap.add_argument("--gzip", action="store_true", help="write .tsv.gz")
    a = ap.parse_args()

    # Why every shard of a rank must be exported together:
    #   * seq is row_number() OVER (PARTITION BY n ORDER BY id). Run per shard,
    #     each shard restarts at 1, so four rows of rank 6 would share seq = 1
    #     and the keyset tiebreak in src/api/cursor.ts stops being unique.
    #   * rank_stats has n as its primary key and every shard carries its own
    #     n = 6 row, so a second shard's COPY is a duplicate-key error.
    # Merging here means one staging table and one global numbering.
    os.makedirs(a.out_dir, exist_ok=True)
    cons = [sqlite3.connect(f"file:{d}?mode=ro", uri=True) for d in a.db]
    have: set = set()
    for c in cons:
        have |= {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    copy_specs: list[tuple[str, list[str], str]] = []
    for table in ORDER:
        if table not in have:
            print(f"  {table}: absent from {a.db}, skipped")
            continue
        cols = TABLES[table]
        ext = ".tsv.gz" if a.gzip else ".tsv"
        path = os.path.join(a.out_dir, table + ext)
        opener = (lambda p: gzip.open(p, "wt", encoding="utf-8", newline="\n")) if a.gzip \
            else (lambda p: open(p, "w", encoding="utf-8", newline="\n"))
        n = 0
        # No ORDER BY: on a TEXT primary key that walks the index and seeks the
        # table once per row (see qmd/bigcell.py). COPY does not care about
        # order; seq is assigned by the staging INSERT's own ORDER BY.
        seen_pk: set = set()
        with opener(path) as fh:
            for con in cons:
                try:
                    cur = con.execute(f"SELECT {', '.join(cols)} FROM {table}")
                except sqlite3.OperationalError:
                    continue
                for row in cur:
                    # rank_stats is keyed by n and every shard carries its own
                    # row for the rank; keep the first and skip the rest.
                    if table == "rank_stats":
                        if row[0] in seen_pk:
                            continue
                        seen_pk.add(row[0])
                    fh.write("\t".join(pg_text(v) for v in row))
                    fh.write("\n")
                    n += 1
        print(f"  {table}: {n:,} rows -> {os.path.basename(path)}")
        src = f"PROGRAM 'gzip -dc {path}'" if a.gzip else f"'{os.path.abspath(path)}'"
        copy_specs.append((table, cols, src))

    # seq is assigned by INSERT ... SELECT out of an UNLOGGED staging table, not
    # by UPDATE. Measured on 8.3 M rows: `UPDATE ... SET seq = row_number()`
    # leaves exactly one dead tuple per row -- MVCC writes a new tuple version
    # for every update -- which took `quivers` from 1,434 MB to 2,835 MB, a
    # 1.98x bloat needing a VACUUM FULL and 2x transient disk. It was also the
    # bulk of an 82-minute load. Staging costs one extra COPY and stays clean.
    # psql's \copy, NOT SQL COPY. SQL `COPY ... FROM '/path'` makes the SERVER
    # read the path, and `COPY ... FROM PROGRAM` makes the server run a shell
    # command; both need superuser and both assume the data sits on the database
    # host. That is true of a local Postgres -- the server is your laptop, which
    # is why the Phase 0 rehearsal passed -- and false of every managed provider,
    # PlanetScale included. \copy is a psql client meta-command: it reads the
    # file (or pipes the program) HERE and streams it over the wire as
    # COPY ... FROM STDIN, which needs no special grant. It must stay on one
    # line; psql meta-commands end at the newline.
    def copy_line(target, cols, src):
        return f"\\copy {target} ({', '.join(cols)}) FROM {src}"

    cmd_path = os.path.join(a.out_dir, "load.sql")
    with open(cmd_path, "w", encoding="utf-8") as fh:
        fh.write("-- Generated by scripts/pg-export.py. Run with psql -f (the \\copy\n"
                 "-- lines are psql client meta-commands, not SQL).\n"
                 "-- Order: drizzle-pg/0001_init.sql -> this file ->\n"
                 "--        drizzle-pg/0002_indexes.sql -> ANALYZE.\n\n")
        for t in ORDER:
            for table, cols, src in copy_specs:
                if table != t:
                    continue
                if t in ("quivers", "mutation_classes"):
                    # Stage, then INSERT with the row number computed in one pass.
                    fh.write(f"CREATE UNLOGGED TABLE stage_{t} (LIKE {t} EXCLUDING ALL);\n")
                    fh.write(copy_line(f"stage_{t}", cols, src) + "\n")
                    fh.write(f"INSERT INTO {t} ({', '.join(TABLES[t])}, seq)\n"
                             f"  SELECT {', '.join(TABLES[t])},\n"
                             f"         row_number() OVER (PARTITION BY n ORDER BY id)\n"
                             f"  FROM stage_{t};\n"
                             f"DROP TABLE stage_{t};\n\n")
                else:
                    fh.write(copy_line(t, cols, src) + "\n")
        fh.write("\n-- (n, seq) is id order per rank, which is what src/api/cursor.ts assumes.\n")
    print(f"  -> {cmd_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
