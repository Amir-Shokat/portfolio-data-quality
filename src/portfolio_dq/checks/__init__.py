"""The check catalogue as code.

Every check is registered with a :class:`CheckSpec` (id, family, within/cross-chain scope,
severity, owner, tolerance) and returns a :class:`CheckResult` holding an exception register.
"""

from __future__ import annotations

import pandas as pd

from ..model import SCOPE_ORDER
from . import family_a, family_b, family_c, family_d  # noqa: F401  (registers the checks)
from .base import REGISTRY, CheckResult, CheckSpec, run_check

__all__ = ["REGISTRY", "CheckResult", "CheckSpec", "catalogue", "run_all", "run_check",
           "summary"]


def _order(spec: CheckSpec) -> tuple:
    return (SCOPE_ORDER.index(spec.scope), spec.family, spec.id)


def catalogue() -> pd.DataFrame:
    """One row per check, in run order: gates, within-chain, cross-chain, reference, ..."""
    specs = sorted((s for s, _ in REGISTRY.values()), key=_order)
    return pd.DataFrame([dict(id=s.id, family=s.family, scope=s.scope_label, title=s.title,
                              severity=s.severity, owner=s.owner, tolerance=s.tolerance,
                              question=s.question) for s in specs])


def run_all(ds, ids=None, **params) -> list[CheckResult]:
    """Run the catalogue (or ``ids``) in order. Parameters are passed to checks that take them."""
    specs = sorted((s for s, _ in REGISTRY.values()), key=_order)
    out = []
    for s in specs:
        if ids and s.id not in ids:
            continue
        fn = REGISTRY[s.id][1]
        names = fn.__code__.co_varnames[1:fn.__code__.co_argcount]
        out.append(run_check(s.id, ds, **{k: v for k, v in params.items() if k in names}))
    return out


def summary(results: list[CheckResult]) -> pd.DataFrame:
    """The manager's layer: one row per check, scope label kept on every row."""
    return pd.DataFrame([r.summary_row() for r in results])
