"""Excel tracker: the three-layer exception register plus the scenario test log.

Sheets:

* ``Summary``: one row per check with its WITHIN/CROSS label (the layer a manager reads);
* ``Exceptions``: one row per exception, worst first within each check;
* ``Catalogue``: the check catalogue (id, family, scope, severity, owner, tolerance);
* ``Scenario tests``: one row per protocol run, with an Outcome drop-down;
* ``Replay differences``: the non-matching rows behind each scenario.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from .checks import catalogue, summary
from .scenarios import OUTCOMES, records_frame

HEADER_FILL = PatternFill("solid", fgColor="1F3A5F")
SCOPE_FILL = {"WITHIN": PatternFill("solid", fgColor="FDE9E7"),
              "CROSS-CHAIN": PatternFill("solid", fgColor="FFF4D6"),
              "VALIDITY": PatternFill("solid", fgColor="E8EEF8")}
OUTCOME_FILL = {"CORRECTLY_DETECTED": "E2F0D9", "NO_DISCREPANCY_AS_EXPECTED": "E2F0D9",
                "MISSED_DISCREPANCY": "F8D7DA", "UNEXPECTED_DISCREPANCY": "F8D7DA",
                "MISCLASSIFIED": "FFF4D6", "EXECUTION_FAILED": "F8D7DA",
                "NOT_A_CANDIDATE": "EEEEEE"}


def _cell(v):
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime().date()
    if isinstance(v, (dict, list, tuple, set, frozenset)):
        return str(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        return str(v)
    if hasattr(v, "item"):
        return v.item()
    return v


def _sheet(wb: Workbook, title: str, df: pd.DataFrame, widths: dict | None = None):
    ws = wb.create_sheet(title)
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append([_cell(v) for v in row])
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = HEADER_FILL
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    for i, col in enumerate(df.columns, start=1):
        sample = [len(str(col))] + [len(str(v)) for v in df[col].head(200)]
        width = (widths or {}).get(col, min(max(sample) + 2, 60))
        ws.column_dimensions[get_column_letter(i)].width = width
    if len(df):
        ws.auto_filter.ref = ws.dimensions
    return ws


def exception_register(results) -> pd.DataFrame:
    """All exception rows of all checks, with the check id and scope label on every row."""
    frames = []
    for r in results:
        e = r.exceptions
        if e is None or not len(e):
            continue
        e = e.copy()
        if "diff" in e:
            e = e.reindex(pd.to_numeric(e["diff"], errors="coerce").abs()
                          .sort_values(ascending=False, na_position="last").index)
        e.insert(0, "scope", r.spec.scope_label)
        e.insert(0, "check", r.spec.id)
        frames.append(e)
    if not frames:
        return pd.DataFrame(columns=["check", "scope"])
    df = pd.concat(frames, ignore_index=True, sort=False)
    front = ["check", "scope", "customer_id", "symbol", "mismatch_type", "diff"]
    return df[[c for c in front if c in df] + [c for c in df if c not in front]]


def write_tracker(path, results=None, records=None) -> Path:
    """Write the tracker workbook. Either part may be omitted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)
    if results is not None:
        ws = _sheet(wb, "Summary", summary(results), widths={"check": 44, "note": 60})
        for row in ws.iter_rows(min_row=2):
            label = str(row[2].value or "")
            for key, fill in SCOPE_FILL.items():
                if label.startswith(key) or key in label:
                    row[2].fill = fill
        _sheet(wb, "Exceptions", exception_register(results))
        _sheet(wb, "Catalogue", catalogue(), widths={"question": 70, "title": 44})
    if records is not None:
        df = records_frame(records)
        ws = _sheet(wb, "Scenario tests", df, widths={"Notes": 60, "Expected report": 50,
                                                      "Actual report": 50})
        col = get_column_letter(list(df.columns).index("Outcome") + 1)
        dv = DataValidation(type="list", formula1='"' + ",".join(OUTCOMES) + '"',
                            allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{max(len(df) + 1, 200)}")
        for row in ws.iter_rows(min_row=2):
            cell = row[list(df.columns).index("Outcome")]
            if cell.value in OUTCOME_FILL:
                cell.fill = PatternFill("solid", fgColor=OUTCOME_FILL[cell.value])
        diffs = [r.report.differences.assign(test=r.scenario.id) for r in records
                 if r.report is not None and len(r.report.differences)]
        diff_df = pd.concat(diffs, ignore_index=True) if diffs else pd.DataFrame(columns=["test"])
        _sheet(wb, "Replay differences", diff_df[["test"] + [c for c in diff_df if c != "test"]])
    wb.save(path)
    return path
