"""
scripts/backfill_labeling_counts.py

One-off (re-runnable) backfill for quivers.labeling_count.

Counts, for every unlabeled quiver, how many labeled matrices across all
mutation classes map to it (the labeled orbit stores one entry per labeling,
keyed by qmd_id), and writes that onto the quiver row.  Populate sets this
column going forward; this catches up existing data after the migration.

Usage:
    python scripts/backfill_labeling_counts.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import func

from qmd import models
from qmd.db import SessionLocal


def main() -> None:
    db = SessionLocal()
    try:
        counts: dict[str, int] = {}
        for mc in db.query(models.MutationClass).all():
            for entry in (mc.labeled_quivers or []):
                qid = entry.get("qmd_id")
                if qid:
                    counts[qid] = counts.get(qid, 0) + 1

        updated = 0
        for q in db.query(models.Quiver).all():
            q.labeling_count = counts.get(q.quiver_id, 1)
            updated += 1
        db.commit()

        total = db.query(func.coalesce(func.sum(models.Quiver.labeling_count), 0)).scalar()
        nulls = db.query(models.Quiver).filter(models.Quiver.labeling_count.is_(None)).count()
        print(f"Updated {updated} quivers; SUM(labeling_count) = {total}; "
              f"remaining NULLs = {nulls}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
