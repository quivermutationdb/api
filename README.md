# QMD API

Backend for the [Quiver Mutation Database](https://quivermutationdb.org).

## Structure

```
api/
├── qmd/
│   ├── core.py          # Matrix types, mutation, ID generation, BFS explorer
│   └── __init__.py
├── app/                 # FastAPI app (coming soon)
│   ├── main.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   └── routers/
├── data/
│   └── seeds/           # Existing seed data (JSON)
├── scripts/
│   └── ingest.py        # Load seed data into PostgreSQL
├── tests/
│   └── test_core.py     # Full test suite for core.py
├── requirements.txt
└── README.md
```

## Identifiers

**Quiver ID:** `Q.n{vertices}.{sha256[:16]}`
- Hashes the labeled exchange matrix (row-major JSON, compact)
- Example: `Q.n4.a3f2c1d9e8b70f21`

**Mutation Class ID:** `MC.n{vertices}.{sha256[:16]}`
- Hashes the lex-min exchange matrix across all bounded-mutation-reachable
  matrices in the class (no vertex relabeling)
- Example: `MC.n4.f8a21c3d7e904b56`

## Generation Rules

- Seed quivers: all skew-symmetric `{0, ±1, ±2}` matrices on ≤ 4 vertices
- Mutation bound: `|b_ij| ≤ 2` at every step
- If a mutation would produce `|b_ij| > 2`, that branch is stopped and the
  class is marked `is_open = True`

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the generation pipeline
python -c "
from qmd.core import run_generation
r = run_generation(max_vertices=4, bound=2)
print(f'{len(r.quivers)} quivers in {len(r.classes)} mutation classes')
"

# Run tests
python -m pytest tests/ -v
```

## Known Results (sanity checks)

| Quiver type | Mutation class size |
|---|---|
| A2          | 2                   |
| A3          | 14                  |
| D4          | 132 (finite type)   |

## License

CC-BY-4.0
