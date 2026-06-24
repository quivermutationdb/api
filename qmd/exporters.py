"""
qmd/exporters.py

Serialise export-ready row dicts (see crud.export_rows) into downloadable
file bytes.  Two formats: CSV (universal) and XLSX (Excel).

Booleans are rendered as TRUE/FALSE and None as an empty cell so the output
is clean in both a text editor and a spreadsheet program.
"""

import csv
import io


def _cell(value):
    """Normalise a Python value for a flat CSV/XLSX cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return value


def to_csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row.get(c)) for c in columns])
    # utf-8-sig so Excel opens accented text / unicode cleanly on double-click.
    return buf.getvalue().encode("utf-8-sig")


def to_xlsx_bytes(rows: list[dict], columns: list[str]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = "quivers"

    ws.append(columns)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"

    for row in rows:
        ws.append([_cell(row.get(c)) for c in columns])

    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()
