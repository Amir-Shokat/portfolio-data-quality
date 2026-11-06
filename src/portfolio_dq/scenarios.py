"""The reconciliation test protocol, as code.

For each scenario:

1. **Query**: run the scenario's evidence SQL and check that the chosen customer is a
   real candidate (not just assumed to be one).
2. **Record the expected discrepancy** before running anything: the set of
   ``(table, status)`` pairs the report should contain, or none for a negative control.
3. **Run the comparison**: :func:`portfolio_dq.replay.rebuild_compare`, report-only.
4. **Compare report vs evidence**: expected pairs against reported pairs.
5. **Record the outcome**: one of :data:`OUTCOMES`, with missing and unexpected pairs.

A scenario may carry a ``tamper``: SQL applied to a disposable copy to create a controlled
discrepancy (the stored data itself is never changed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources

import pandas as pd

from .generator import DEFECTS, make_dataset
from .replay import rebuild_compare

OUTCOMES = ("CORRECTLY_DETECTED", "NO_DISCREPANCY_AS_EXPECTED", "MISSED_DISCREPANCY",
            "UNEXPECTED_DISCREPANCY", "MISCLASSIFIED", "EXECUTION_FAILED", "NOT_A_CANDIDATE")

#: the scenario dataset carries every defect except the lost lineage link, which S12
#: applies itself (otherwise it would hide S07 and S10 behind missing rebuild rows)
SCENARIO_DEFECTS = [d for d in DEFECTS if d != "unlinked_isin_change"]


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    sql: str                     # file name under portfolio_dq/sql
    customer: str
    symbols: tuple
    expected: frozenset          # {(table, status)}; empty for a negative control
    tamper: str = ""
    note: str = ""

    def query(self) -> str:
        return resources.files("portfolio_dq").joinpath("sql", self.sql).read_text("utf-8")


def _exp(*pairs) -> frozenset:
    return frozenset(pairs)


SCENARIOS = [
    Scenario("S01", "Simple trade baseline (negative control)", "S01_simple_trade_baseline.sql",
             "C001", ("EPSIL",), _exp(),
             note="run first: a difference here is a fault in the rebuild"),
    Scenario("S02", "Asset mismatch between snapshots", "S02_asset_mismatch.sql", "C002",
             ("ALPHA",), _exp(("positions_latest", "MISMATCH")),
             tamper="UPDATE positions_latest SET quantity = quantity + 150 "
                    "WHERE customer_id = 'C002' AND symbol = 'ALPHA';",
             note="controlled discrepancy on a disposable copy"),
    Scenario("S03", "Admin registration", "S03_admin_registration.sql", "C011", ("DELTA",),
             _exp(), note="admin rows pass through; the plug is reported by check A1, not here"),
    Scenario("S04", "Cash dividend", "S04_cash_dividend.sql", "C024", ("ALPHA",),
             _exp(("position_events", "MISSING_STORED"), ("positions_daily", "MISSING_STORED"))),
    Scenario("S05", "Bonus shares", "S05_bonus_shares.sql", "C003", ("BETA",), _exp()),
    Scenario("S06", "Rights lifecycle", "S06_rights_lifecycle.sql", "C004",
             ("GAMMA", "GAMMA_R"), _exp(), note="select the main and the rights symbol together"),
    Scenario("S07", "Customer leaving and returning", "S07_leave_and_return.sql", "C005",
             ("DELTA",), _exp(("depository_daily", "MISMATCH")),
             note="no check fires on this unit; only the rebuild sees the row type"),
    Scenario("S08", "Same-day ISIN change", "S08_same_day_isin_change.sql", "C007", ("BETA",),
             _exp()),
    Scenario("S09", "Uncalculated rows", "S09_uncalculated_rows.sql", "C026", ("EPSIL",),
             _exp(("position_events", "MISMATCH"), ("positions_daily", "MISSING_STORED"),
                  ("positions_latest", "MISMATCH")),
             note="queue lag, not drift: the rebuild calculates what storage has not yet"),
    Scenario("S10", "Historical vs current ISIN", "S10_historical_isin.sql", "C028", ("DELTA",),
             _exp()),
    Scenario("S11", "Soft-deleted rows (negative control)", "S11_soft_deleted_rows.sql", "C010",
             ("EPSIL",), _exp()),
    Scenario("S12", "Lineage link lost", "S12_lineage_link_lost.sql", "C028", ("DELTA",),
             _exp(("depository_daily", "MISSING_REBUILT"),
                  ("depository_latest", "MISSING_REBUILT")),
             tamper="DELETE FROM isin_lineage;"),
    Scenario("S13", "One security under two symbol keys", "S13_split_symbol_key.sql", "C016",
             ("ALPHA", "ALPHA_V"),
             _exp(("depository_daily", "MISSING_STORED"), ("depository_daily", "MISSING_REBUILT"),
                  ("depository_latest", "MISSING_STORED"),
                  ("depository_latest", "MISSING_REBUILT"))),
    Scenario("S14", "Informational field only", "S14_informational_field.sql", "C027",
             ("BETA",), _exp(("depository_daily", "MISMATCH")),
             note="severity must be INFO and the unit must count as matched"),
]


@dataclass
class TestRecord:
    scenario: Scenario
    candidate: bool
    evidence_rows: int
    expected: frozenset
    reported: frozenset
    outcome: str
    report: object = None
    error: str = ""
    details: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        s = self.scenario
        fmt = lambda pairs: "; ".join(f"{t}:{st}" for t, st in sorted(pairs)) or "none"  # noqa
        missing, extra = self.expected - self.reported, self.reported - self.expected
        c = self.report.counters if self.report is not None else {}
        return {"Test ID": s.id, "Scenario": s.title, "Evidence query": s.sql,
                "Customer-symbol": f"{s.customer} / {','.join(s.symbols)}",
                "Candidate confirmed": "yes" if self.candidate else "no",
                "Evidence rows": self.evidence_rows, "Tamper applied": s.tamper or "none",
                "Expected report": fmt(self.expected), "Actual report": fmt(self.reported),
                "Units with error": c.get("units_with_error"),
                "Info rows": int((self.report.differences["severity"] == "INFO").sum())
                if self.report is not None else None,
                "Outcome": self.outcome,
                "Missed": fmt(missing) if missing else "", "Unexpected": fmt(extra) if extra else "",
                "Notes": "; ".join(x for x in (s.note, self.error) if x)}


def judge(expected: frozenset, reported: frozenset) -> str:
    if not expected and not reported:
        return "NO_DISCREPANCY_AS_EXPECTED"
    if expected == reported:
        return "CORRECTLY_DETECTED"
    if expected and not reported:
        return "MISSED_DISCREPANCY"
    if not expected:
        return "UNEXPECTED_DISCREPANCY"
    return "MISCLASSIFIED"


def run_scenario(ds, s: Scenario) -> TestRecord:
    try:
        work = ds.with_sql(s.tamper) if s.tamper else ds
        evidence = work.sql(s.query())                       # 1. query
        candidate = (evidence["customer_id"] == s.customer).any() if len(evidence) else False
        if not candidate:
            return TestRecord(s, False, len(evidence), s.expected, frozenset(), "NOT_A_CANDIDATE")
        report = rebuild_compare(work, s.customer, s.symbols)   # 3. run (report-only)
        d = report.differences
        reported = frozenset(zip(d["table"], d["status"]))
        return TestRecord(s, True, int((evidence["customer_id"] == s.customer).sum()),
                          s.expected, reported, judge(s.expected, reported), report)
    except Exception as exc:
        return TestRecord(s, False, 0, s.expected, frozenset(), "EXECUTION_FAILED",
                          error=f"{exc.__class__.__name__}: {exc}")


def run_scenarios(ds=None, ids=None, seed: int = 7) -> list[TestRecord]:
    """Run the protocol over every scenario (or ``ids``) on the scenario dataset."""
    ds = ds if ds is not None else make_dataset(seed, SCENARIO_DEFECTS)
    return [run_scenario(ds, s) for s in SCENARIOS if not ids or s.id in ids]


def records_frame(records: list[TestRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.as_row() for r in records])
