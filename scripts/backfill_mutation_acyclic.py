"""
scripts/backfill_mutation_acyclic.py

Re-resolve quivers' is_mutation_acyclic on existing data using the subquiver
fallback, without re-running the (expensive) local-acyclicity search.

For each mutation class it recomputes the BFS base state from the stored
labeled orbit, then, in ascending rank order, upgrades an "unknown" (open
class, no acyclic member) to False when a member has an induced subquiver
already known to be not mutation-acyclic (Markov is the rank-3 base case).

Only None -> False changes are possible, so this never overwrites a proved
value.  Re-runnable.

Usage:
    python scripts/backfill_mutation_acyclic.py            # apply
    python scripts/backfill_mutation_acyclic.py --dry-run  # report only
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qmd import crud, invariants, models
from qmd.core import to_matrix
from qmd.db import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill is_mutation_acyclic.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        class_infos = []
        for mc in db.query(models.MutationClass).all():
            matrices = [to_matrix(e["matrix"]) for e in (mc.labeled_quivers or [])]
            qids = [e["qmd_id"] for e in (mc.labeled_quivers or [])]
            base = invariants.class_is_mutation_acyclic(matrices, mc.is_open)
            class_infos.append((mc.mc_id, mc.n_vertices, base, matrices, qids))

        resolved = crud._resolve_mutation_acyclic(class_infos)

        changed = 0
        for mc in db.query(models.MutationClass).all():
            new = resolved.get(mc.mc_id)
            if mc.is_mutation_acyclic != new:
                print(f"  {mc.mc_id}  rank {mc.n_vertices}  "
                      f"{mc.is_mutation_acyclic!r} -> {new!r}")
                if not args.dry_run:
                    mc.is_mutation_acyclic = new
                changed += 1

        if args.dry_run:
            db.rollback()
            print(f"[dry-run] would change {changed} classes")
        else:
            db.commit()
            print(f"Updated {changed} classes")
    finally:
        db.close()


if __name__ == "__main__":
    main()
