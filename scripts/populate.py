"""
scripts/populate.py

One-off (re-runnable) data loader for the Quiver Mutation Database.

Two modes:

Postgres (legacy, default)
    Runs the full generation pipeline and writes the results to the database
    pointed to by DATABASE_URL.  Safe to re-run — uses upsert semantics, so
    re-running with the same parameters is idempotent.

D1 export (Cloudflare migration)
    --export-d1 DIR emits one self-contained SQL file per rank
    (DIR/qmd-n{k}.sql) for `wrangler d1 execute qmd --remote --file=...`,
    plus a manifest and per-rank checkpoints so an interrupted run resumes
    where it left off.  Needs no database driver or DATABASE_URL — it runs
    on a bare generation box.

Usage:
    python scripts/populate.py                          # Postgres, n<=4, bound=2
    python scripts/populate.py --max-vertices 4 --bound 2
    python scripts/populate.py --export-d1 dist/d1      # SQL files for D1
    python scripts/populate.py --export-d1 dist/d1 --ranks 4 --force
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate the QMD database.")
    parser.add_argument("--max-vertices", type=int, default=4)
    parser.add_argument("--bound", type=int, default=2)
    parser.add_argument("--export-d1", metavar="DIR", default=None,
                        help="Write per-rank SQL files for wrangler d1 execute "
                             "instead of writing to Postgres.")
    parser.add_argument("--ranks", default=None,
                        help="Comma-separated ranks to export (default: all "
                             "up to --max-vertices). Export mode only.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate ranks even if their exported file is "
                             "up to date. Export mode only.")
    args = parser.parse_args()

    if args.export_d1:
        from qmd.d1_export import export_ranks
        ranks = ([int(r) for r in args.ranks.split(",")] if args.ranks
                 else None)
        print(f"Exporting D1 SQL to {args.export_d1} "
              f"(ranks {ranks or f'1..{args.max_vertices}'}, "
              f"bound |b_ij| <= {args.bound}) ...")
        export_ranks(args.export_d1, max_vertices=args.max_vertices,
                     bound=args.bound, ranks=ranks, force=args.force)
        print("Done.")
        return

    from qmd.core import run_generation
    from qmd.crud import upsert_generation_result
    from qmd.db import SessionLocal

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
