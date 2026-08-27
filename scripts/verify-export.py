#!/usr/bin/env python3
"""
scripts/verify-export.py DIR [--ranks a,b]

Gate before a production load: re-parse every quiver row in the export
parts and refuse the release if any quiver is disconnected, if a quiver's
stored is_connected flag disagrees with its matrix, if a labeling matrix is
disconnected, or if a rank's quiver count differs from the manifest.
Exit status 1 on any problem.
"""
import argparse
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qmd.core import is_connected  # noqa: E402
from qmd.encoding import decode_upper  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MIGRATIONS = os.path.join(ROOT, "drizzle")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ranks", default="")
    args = ap.parse_args()
    manifest = json.load(open(os.path.join(args.dir, "manifest.json")))
    want = {int(x) for x in args.ranks.split(",") if x}
    problems = 0
    for n_str, entry in sorted(manifest["ranks"].items(), key=lambda kv: int(kv[0])):
        n = int(n_str)
        if want and n not in want:
            continue
        con = sqlite3.connect(":memory:")
        for name in sorted(os.listdir(MIGRATIONS)):
            if name.endswith(".sql"):
                con.executescript(open(os.path.join(MIGRATIONS, name), encoding="utf-8").read()
                                  .replace("--> statement-breakpoint", ""))
        for part in entry["parts"]:
            with open(os.path.join(args.dir, part["file"]), encoding="utf-8") as f:
                con.executescript(f.read())
        total = con.execute("SELECT count(*) FROM quivers WHERE n=?", (n,)).fetchone()[0]
        disc = 0
        flag_mismatch = 0
        for qid, upper, flag in con.execute("SELECT id, exchange_matrix, is_connected FROM quivers WHERE n=?", (n,)):
            c = is_connected(decode_upper(n, upper))
            if not c:
                disc += 1
            if bool(flag) != c:
                flag_mismatch += 1
        lab_disc = sum(1 for (upper,) in con.execute(
            "SELECT l.matrix FROM labelings l JOIN mutation_classes m ON m.id = l.mutation_class_id WHERE m.n=?", (n,))
            if not is_connected(decode_upper(n, upper)))
        ok = disc == 0 and flag_mismatch == 0 and lab_disc == 0 and total == entry.get("quiver_count", total)
        print(f"  rank {n}: {total} quivers, disconnected={disc}, flag mismatches={flag_mismatch}, "
              f"disconnected labelings={lab_disc}, manifest count={entry.get('quiver_count')} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            problems += 1
    print("verify-export:", "OK" if problems == 0 else f"{problems} rank(s) FAILED")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
