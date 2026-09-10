#!/usr/bin/env python3
"""
Print files listed in the OLD corpus manifest but not the NEW one.

Used by scripts/r2-upload-corpus.sh to prune objects a release leaves behind.
This is needed because a rank can change SHAPE between releases: rank 6 outgrew
the 315 MB single-object ceiling and became numbered parts, and a leftover
qmd-n6.ndjson.gz would otherwise stay in the bucket, be served, and be wrong.
There is no `wrangler r2 object list`, so the published manifest is the only
inventory of what is currently up.

Prints nothing when the old manifest is missing or unreadable -- a first
publish, or a corrupt download, should never be read as "delete everything".
"""
import json
import sys


def names(path: str) -> set[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
    except Exception:
        return set()
    if isinstance(m.get("files"), list):
        return set(m["files"])
    # Manifests written before ranks could be split named one file per rank.
    out = set()
    for r in m.get("ranks", []):
        for part in r.get("parts", [r]):
            if isinstance(part, dict) and part.get("file"):
                out.add(part["file"])
    for entry in (m.get("aux") or {}).values():
        if isinstance(entry, dict) and entry.get("file"):
            out.add(entry["file"])
    return out


def main() -> int:
    old, new = names(sys.argv[1]), names(sys.argv[2])
    if not new:
        print("refusing to prune: the new manifest lists no files", file=sys.stderr)
        return 1
    for f in sorted(old - new):
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
