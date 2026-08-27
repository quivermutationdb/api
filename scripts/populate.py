"""
scripts/populate.py

Generate the dataset and export it as per-rank SQL files for Cloudflare D1.

Runs the full generation pipeline (qmd/core.py) and writes one
per-rank set of SQL parts (DIR/qmd-n{k}.NNN.sql) for

    scripts/import-d1.sh DIR [--remote]        # parts in order, ranks ascending

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
    parser.add_argument("--node-cap", type=int, default=None,
                        help="Stop a class BFS after this many labeled matrices; "
                             "such classes are stored as exploration='truncated' "
                             "with unknown finiteness (never as finite).")
    parser.add_argument("--part-bytes", type=int, default=None,
                        help="Split each rank's SQL into parts of at most this "
                             "many bytes (default 64 MB).")
    parser.add_argument("--generator", choices=["orderly", "brute", "sample"],
                        default="orderly",
                        help="Phase-1 seeds: orderly generation (exact census, "
                             "default), brute-force enumeration (tiny cells), or "
                             "a uniform sample of labeled matrices (--sample N).")
    parser.add_argument("--sample", type=int, default=None,
                        help="With --generator sample: number of distinct quivers "
                             "to draw per rank.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="Process-pool size for generation, BFS and invariants "
                             "(default: CPUs - 2).")
    parser.add_argument("--count-only", action="store_true",
                        help="Print the exact cell sizes (Burnside) and exit.")
    args = parser.parse_args()

    if args.count_only:
        from qmd.census import count_quivers
        ranks_ = [int(r) for r in args.ranks.split(",")] if args.ranks else range(1, args.max_vertices + 1)
        for n in ranks_:
            print(f"  n={n:<2} |b_ij|<={args.bound}: {count_quivers(n, args.bound):,} unlabeled quivers")
        return

    from qmd.d1_export import export_ranks
    ranks = [int(r) for r in args.ranks.split(",")] if args.ranks else None
    print(f"Exporting D1 SQL to {args.export_d1} "
          f"(ranks {ranks or f'1..{args.max_vertices}'}, "
          f"bound |b_ij| <= {args.bound}) ...")
    kwargs = {}
    if args.part_bytes:
        kwargs["part_bytes"] = args.part_bytes
    export_ranks(args.export_d1, max_vertices=args.max_vertices,
                 bound=args.bound, ranks=ranks, force=args.force,
                 node_cap=args.node_cap, generator=args.generator,
                 sample=args.sample, sample_seed=args.sample_seed,
                 workers=args.workers, **kwargs)
    print("Done.")


if __name__ == "__main__":
    main()
