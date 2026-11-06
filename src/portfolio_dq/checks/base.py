"""Check specification, result object and registry."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from ..model import SCOPE_LABELS


@dataclass(frozen=True)
class CheckSpec:
    """Static description of one check in the catalogue."""
    id: str
    title: str
    family: str          # A, B, C, D (gates are family A checks run first)
    scope: str           # key of model.SCOPE_LABELS
    severity: str        # critical, high, medium, low, info
    owner: str
    question: str
    tolerance: str = "exact"
    implementation: str = "pandas"

    @property
    def scope_label(self) -> str:
        return SCOPE_LABELS[self.scope]


@dataclass
class CheckResult:
    """Outcome of one check: an exception register plus context.

    ``exceptions`` holds one row per exception at the grain being checked, with a
    ``mismatch_type`` and, where it makes sense, a signed ``diff`` (A - B). Rows that are
    worth showing but are not exceptions (within tolerance, known variants, backlog) go to
    ``observations``.
    """
    spec: CheckSpec
    exceptions: pd.DataFrame
    denominator: int = 0
    observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    status: str = ""

    def __post_init__(self):
        if self.exceptions is None:
            self.exceptions = pd.DataFrame()
        if not self.status:
            if self.spec.scope == "profile":
                self.status = "PROFILE"
            else:
                self.status = "FAIL" if len(self.exceptions) else "PASS"

    @property
    def fired(self) -> bool:
        return self.status == "FAIL"

    @property
    def n_exceptions(self) -> int:
        return len(self.exceptions)

    @property
    def customers_affected(self) -> int:
        e = self.exceptions
        return int(e["customer_id"].dropna().nunique()) if "customer_id" in e else 0

    @property
    def magnitude(self) -> float:
        e = self.exceptions
        return float(e["diff"].abs().sum()) if "diff" in e and len(e) else 0.0

    @property
    def worst(self) -> float:
        e = self.exceptions
        return float(e["diff"].abs().max()) if "diff" in e and len(e) else 0.0

    @property
    def rate(self) -> float | None:
        return self.n_exceptions / self.denominator if self.denominator else None

    def summary_row(self) -> dict:
        s = self.spec
        return dict(id=s.id, check=s.title, scope=s.scope_label, family=s.family,
                    status=self.status, exceptions=self.n_exceptions,
                    customers=self.customers_affected, denominator=self.denominator,
                    rate=None if self.rate is None else round(self.rate, 4),
                    magnitude=round(self.magnitude, 2), worst=round(self.worst, 2),
                    observations=len(self.observations), severity=s.severity, owner=s.owner,
                    tolerance=s.tolerance, note="; ".join(self.notes))


REGISTRY: dict[str, tuple[CheckSpec, Callable]] = {}


def check(**spec_kwargs):
    """Register a check function ``f(ds, **params) -> CheckResult`` under its spec."""
    spec = CheckSpec(**spec_kwargs)

    def wrap(fn):
        REGISTRY[spec.id] = (spec, fn)
        fn.spec = spec
        return fn
    return wrap


def run_check(check_id: str, ds, **params) -> CheckResult:
    spec, fn = REGISTRY[check_id]
    try:
        return fn(ds, **params)
    except Exception as exc:                      # a broken check must not hide the others
        return CheckResult(spec, pd.DataFrame(), status="ERROR",
                           notes=[f"{exc.__class__.__name__}: {exc}",
                                  traceback.format_exc(limit=2)])


def frame(rows, columns) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
