"""
main.py

FastAPI application entry point for the Quiver Mutation Database API.

Run locally with:
    uvicorn main:app --reload

The database is populated offline via scripts/populate.py — this service is
read-only.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session

from qmd import crud, exporters, schemas
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
    representation_type: Optional[str] = None,
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
        "representation_type": representation_type,
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
# Export / download
# ---------------------------------------------------------------------------

_EXPORT_MEDIA = {
    "csv":  "text/csv; charset=utf-8",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def _client_ip(request: Request) -> Optional[str]:
    """Real client IP, honouring the proxy chain Render/Neon sit behind."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


@app.get("/export")
def export(
    request: Request,
    format: str = "csv",
    email: Optional[str] = None,
    name: Optional[str] = None,
    rank: Optional[int] = None,
    dynkin_type: Optional[str] = None,
    representation_type: Optional[str] = None,
    max_edge: Optional[int] = None,
    is_open: Optional[bool] = None,
    orbit_min: Optional[int] = None,
    orbit_max: Optional[int] = None,
    is_acyclic: Optional[bool] = None,
    is_connected: Optional[bool] = None,
    is_simply_laced: Optional[bool] = None,
    is_mutation_finite: Optional[bool] = None,
    db: Session = Depends(get_db),
):
    """
    Download the current cut of the dataset as CSV or Excel.

    Accepts the same filters as /search (omit all for the full dataset) and
    logs the download (cut, format, row count, optional self-reported email).
    """
    fmt = format.lower()
    if fmt not in _EXPORT_MEDIA:
        raise HTTPException(status_code=400, detail="format must be 'csv' or 'xlsx'")

    filters = {
        "rank": rank,
        "dynkin_type": dynkin_type,
        "representation_type": representation_type,
        "max_edge": max_edge,
        "is_open": is_open,
        "orbit_min": orbit_min,
        "orbit_max": orbit_max,
        "is_acyclic": is_acyclic,
        "is_connected": is_connected,
        "is_simply_laced": is_simply_laced,
        "is_mutation_finite": is_mutation_finite,
    }
    rows = crud.export_rows(db, filters=filters)

    if fmt == "csv":
        body = exporters.to_csv_bytes(rows, crud.EXPORT_COLUMNS)
    else:
        body = exporters.to_xlsx_bytes(rows, crud.EXPORT_COLUMNS)

    # Best-effort logging: a tracking failure must never break the download.
    try:
        crud.log_download(
            db,
            fmt=fmt,
            row_count=len(rows),
            filters={k: v for k, v in filters.items() if v is not None},
            email=email,
            name=name,
            ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
            referer=request.headers.get("referer"),
        )
    except Exception:
        db.rollback()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"qmd-quivers-{stamp}.{fmt}"
    return Response(
        content=body,
        media_type=_EXPORT_MEDIA[fmt],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Mutation classes
# ---------------------------------------------------------------------------

@app.get("/classes/{mc_id}", response_model=schemas.ClassDetail)
def get_class(mc_id: str, db: Session = Depends(get_db)):
    row = crud.get_class_detail(db, mc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Mutation class not found")
    return row
