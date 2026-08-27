"""
scripts/populate.py

Generate the dataset and export it as per-rank SQL files for Cloudflare D1.

Runs the full generation pipeline (qmd/core.py) and writes one
self-contained SQL file per rank (DIR/qmd-n{k}.sql) for

    npx wrangler d1 execute qmd --remote --file=DIR/qmd-n{k}.sql

plus a manifest and per-rank checkpoints so an interrupted run resumes
where it left off (see qmd/d1_export.py). Pure Python, no database
driver needed — it runs on a bare generation box.

Usage:
    python scripts/populate.py --export-d1 dist/d1                # n<=4, bound=2
    python scripts/populate.py --export-d1 dist/d1 --ranks 4 --force

(The former direct-to-database write mode was removed in the Cloudflare
migration; regenerate + re-import into D1 instead.)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the QMD dataset and export it as D1 SQL files.")
    parser.add_argument("--max-vertices", type=int, default=4)
    parser.add_argument("--bound", type=int, default=2)
    parser.add_argument("--export-d1", metavar="DIR", required=True,
                        help="Output directory for the per-rank SQL files.")
    parser.add_argument("--ranks", default=None,
                        help="Comma-separated ranks to export (default: all "
                             "up to --max-vertices).")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate ranks even if their exported file is "
                             "up to date.")
    args = parser.parse_args()

    from qmd.d1_export import export_ranks
    ranks = [int(r) for r in args.ranks.split(",")] if args.ranks else None
    print(f"Exporting D1 SQL to {args.export_d1} "
          f"(ranks {ranks or f'1..{args.max_vertices}'}, "
          f"bound |b_ij| <= {args.bound}) ...")
    export_ranks(args.export_d1, max_vertices=args.max_vertices,
                 bound=args.bound, ranks=ranks, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
