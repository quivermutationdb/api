"""
scripts/populate.py

One-off (re-runnable) data loader for the Quiver Mutation Database.

Runs the full generation pipeline and writes the results to the database
pointed to by DATABASE_URL.  Safe to re-run — uses upsert semantics, so
re-running with the same parameters is idempotent.

Usage:
    python scripts/populate.py                          # max_vertices=4, bound=2
    python scripts/populate.py --max-vertices 4 --bound 2
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmd.core import run_generation
from qmd.crud import upsert_generation_result
from qmd.db import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate the QMD database.")
    parser.add_argument("--max-vertices", type=int, default=4)
    parser.add_argument("--bound", type=int, default=2)
    args = parser.parse_args()

    print(f"Generating quivers up to {args.max_vertices} vertices "
          f"(bound |b_ij| <= {args.bound}) ...")
    result = run_generation(max_vertices=args.max_vertices, bound=args.bound)
    n_closed = sum(1 for mc in result.classes.values() if not mc.is_open)
    print(f"  {len(result.quivers)} quivers in {len(result.classes)} mutation "
          f"classes ({n_closed} finite-type, {len(result.classes) - n_closed} open)")

    print("Writing to database ...")
    db = SessionLocal()
    try:
        upsert_generation_result(db, result)
    finally:
        db.close()

    print("Done.")


if __name__ == "__main__":
    main()
