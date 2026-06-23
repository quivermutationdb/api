"""
main.py

FastAPI application entry point for the Quiver Mutation Database API.

Run locally with:
    uvicorn main:app --reload

The database is populated offline via scripts/populate.py — this service is
read-only.
"""

from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from qmd import crud, schemas
from qmd.db import get_db

app = FastAPI(title="Quiver Mutation Database API", version="0.2.0")

# Read-only public API: allow the site (and localhost during development) to
# call it from the browser.  Widen allow_origins to ["*"] if you want the API
# usable from arbitrary third-party web pages.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://quivermutationdb.org",
        "https://www.quivermutationdb.org",
    ],
    allow_origin_regex=r"https?://localhost(:\d+)?",
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "Quiver Mutation Database API",
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/quivers", "/quivers/{id}", "/search", "/classes/{id}"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Quivers
# ---------------------------------------------------------------------------

@app.get("/quivers", response_model=schemas.QuiverListResponse)
def list_quivers(
    rank: Optional[int] = None,
    is_open: Optional[bool] = None,
    dynkin_type: Optional[str] = None,
    sort: str = "num_vertices",
    dir: str = "asc",
    offset: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    filters = {"rank": rank, "is_open": is_open, "dynkin_type": dynkin_type}
    items, total = crud.list_quivers(
        db, filters=filters, sort=sort, direction=dir, offset=offset, limit=limit
    )
    return {"items": items, "total": total}


@app.get("/quivers/{quiver_id}", response_model=schemas.QuiverDetail)
def get_quiver(quiver_id: str, db: Session = Depends(get_db)):
    row = crud.get_quiver_detail(db, quiver_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Quiver not found")
    return row


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.get("/search", response_model=schemas.QuiverListResponse)
def search(
    rank: Optional[int] = None,
    dynkin_type: Optional[str] = None,
    max_edge: Optional[int] = None,
    is_open: Optional[bool] = None,
    orbit_min: Optional[int] = None,
    orbit_max: Optional[int] = None,
    is_acyclic: Optional[bool] = None,
    is_connected: Optional[bool] = None,
    is_simply_laced: Optional[bool] = None,
    is_mutation_finite: Optional[bool] = None,
    offset: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    filters = {
        "rank": rank,
        "dynkin_type": dynkin_type,
        "max_edge": max_edge,
        "is_open": is_open,
        "orbit_min": orbit_min,
        "orbit_max": orbit_max,
        "is_acyclic": is_acyclic,
        "is_connected": is_connected,
        "is_simply_laced": is_simply_laced,
        "is_mutation_finite": is_mutation_finite,
    }
    items, total = crud.list_quivers(db, filters=filters, offset=offset, limit=limit)
    return {"items": items, "total": total}


# ---------------------------------------------------------------------------
# Mutation classes
# ---------------------------------------------------------------------------

@app.get("/classes/{mc_id}", response_model=schemas.ClassDetail)
def get_class(mc_id: str, db: Session = Depends(get_db)):
    row = crud.get_class_detail(db, mc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Mutation class not found")
    return row
