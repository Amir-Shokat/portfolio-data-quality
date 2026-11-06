"""Family B: coverage, freshness and pipeline health."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import model as m
from .base import CheckResult, check, frame
from .common import (Prices, Series, chain1_rows, chain2_rows, dedupe_daily, live_events,
                     unit_series)


def _held_now(ds) -> pd.DataFrame:
    """Positions held at the as-of date (the route that feeds the app)."""
    latest = ds["positions_latest"].drop_duplicates(["customer_id", "symbol"], keep="last")
    return latest[latest["quantity"] > 0]


@check(id="B1", title="Price coverage of held symbols", family="B", scope="reference",
       severity="critical", owner="Market data",
       question="Does every held symbol have a price on or before the as-of date?",
       tolerance="exact; weighted by count and by cost value")
def b1_price_coverage(ds) -> CheckResult:
    spec = b1_price_coverage.spec
    px = Prices(ds["prices"])
    held = _held_now(ds)
    rows = []
    for r in held.itertuples(index=False):
        if px.close(r.symbol, ds.as_of) is None:
            cost = float(r.quantity * r.avg_cost)
            rows.append(dict(customer_id=r.customer_id, symbol=r.symbol, quantity=r.quantity,
                             cost_value=cost, diff=cost, mismatch_type="NO_PRICE"))
    out = frame(rows, ["customer_id", "symbol", "quantity", "cost_value", "diff", "mismatch_type"])
    total = float((held["quantity"] * held["avg_cost"]).sum())
    missing = float(out["cost_value"].sum()) if len(out) else 0.0
    return CheckResult(spec, out, denominator=len(held),
                       metrics=dict(coverage_by_count=round(1 - len(out) / max(len(held), 1), 4),
                                    coverage_by_value=round(1 - missing / total, 4) if total else None))


@check(id="B2", title="Price staleness", family="B", scope="reference", severity="high",
       owner="Market data", question="How old is the last price of each held symbol?",
       tolerance="3 trading days")
def b2_price_staleness(ds, stale_days: int = 3) -> CheckResult:
    spec = b2_price_staleness.spec
    px, cal = Prices(ds["prices"]), ds.calendar
    held = _held_now(ds)
    rows, ages, weights = [], [], []
    for r in held.itertuples(index=False):
        last = px.last_date(r.symbol, ds.as_of)
        if last is None:
            continue                      # no price at all is B1's finding
        age = cal.between(last, ds.as_of)
        value = float(r.quantity * px.close(r.symbol, ds.as_of))
        ages.append(age)
        weights.append(value)
        if age > stale_days:
            rows.append(dict(customer_id=r.customer_id, symbol=r.symbol, last_price_date=last,
                             age_days=age, value=value, diff=age, mismatch_type="STALE_PRICE"))
    metrics = {}
    if ages:
        a = np.array(ages)
        order = np.argsort(a)
        cw = np.cumsum(np.array(weights)[order]) / max(sum(weights), 1e-9)
        metrics = dict(p95_days=float(np.percentile(a, 95)), p99_days=float(np.percentile(a, 99)),
                       value_weighted_p95_days=float(a[order][np.searchsorted(cw, 0.95)]))
    out = frame(rows, ["customer_id", "symbol", "last_price_date", "age_days", "value", "diff",
                       "mismatch_type"])
    return CheckResult(spec, out, denominator=len(ages), metrics=metrics)


@check(id="B3", title="Price plausibility", family="B", scope="reference", severity="medium",
       owner="Market data",
       question="Are prices positive and within the daily limit of the adjusted base price?",
       tolerance="daily limit per instrument class, after corporate-action adjustment")
def b3_price_plausibility(ds) -> CheckResult:
    spec = b3_price_plausibility.spec
    klass = dict(zip(ds["symbols"]["symbol"], ds["symbols"]["instrument_class"]))
    limit = dict(zip(ds["fee_schedule"]["instrument_class"], ds["fee_schedule"]["daily_limit"]))
    cas = ds["corporate_actions"]
    actions = {}
    for ca in cas.itertuples(index=False):
        actions.setdefault((ca.symbol, ca.record_date), []).append((ca.action_type, ca.value))
    rows, obs, n = [], [], 0
    for sym, g in ds["prices"].sort_values("date").groupby("symbol"):
        prev, frozen = None, 0
        for r in g.itertuples(index=False):
            n += 1
            if r.close <= 0:
                rows.append(dict(symbol=sym, date=r.date, close=r.close, base=prev, move=None,
                                 limit=None, diff=None, mismatch_type="NON_POSITIVE_PRICE"))
            elif prev is not None:
                base = m.adjusted_base_price(prev, actions.get((sym, r.date), []))
                move = r.close / base - 1
                lim = limit.get(klass.get(sym), 0.05)
                if abs(move) > lim + 1e-9:
                    rows.append(dict(symbol=sym, date=r.date, close=r.close, base=round(base, 2),
                                     move=round(move, 4), limit=lim, diff=round(move, 4),
                                     mismatch_type="LIMIT_BREAK"))
                frozen = frozen + 1 if r.close == prev else 0
                if frozen == 5:
                    obs.append(dict(symbol=sym, date=r.date, note="price unchanged for 5 days"))
            prev = r.close
    out = frame(rows, ["symbol", "date", "close", "base", "move", "limit", "diff", "mismatch_type"])
    return CheckResult(spec, out, denominator=n, observations=pd.DataFrame(obs))


@check(id="B4", title="Calculation backlog", family="B", scope="within_chain2", severity="high",
       owner="Engineering (ledger)",
       question="Are there ledger rows that were never calculated (pending or failed)?",
       tolerance="exact")
def b4_backlog(ds) -> CheckResult:
    spec = b4_backlog.spec
    ev = live_events(ds)
    bad = ev[ev["calc_status"] != m.CALCULATED].copy()
    bad["diff"] = [m.signed_quantity(t, q) for t, q in zip(bad["event_type"], bad["quantity"])]
    bad["mismatch_type"] = np.where(bad["calc_status"] == m.FAILED, "FAILED", "NOT_CALCULATED")
    out = bad[["customer_id", "symbol", "date", "time", "event_type", "calc_status", "diff",
               "mismatch_type"]].reset_index(drop=True)
    return CheckResult(spec, out, denominator=len(ev),
                       notes=["a pending row is queue lag; a failed row is a real error"])


@check(id="B6", title="Position update recency per route", family="B", scope="cross_chain",
       severity="high", owner="Data reconciliation",
       question="Has one route stopped updating while the other moved on?",
       tolerance="2 trading days")
def b6_recency(ds, lag_days: int = 2) -> CheckResult:
    spec = b6_recency.spec
    cal = ds.calendar
    dep, pos = unit_series(chain1_rows(ds)), unit_series(chain2_rows(ds))
    rows = []
    for unit in sorted(set(dep) | set(pos), key=str):
        a, b = dep.get(unit, Series([])), pos.get(unit, Series([]))
        da, db = (a.change_dates() or [None])[-1], (b.change_dates() or [None])[-1]
        qa, qb = a.at(cal.as_of), b.at(cal.as_of)
        if da is None or db is None or qa == qb:
            continue
        gap = cal.between(min(da, db), max(da, db))
        if gap > lag_days:
            rows.append(dict(customer_id=unit[0], canonical=unit[1], last_change_depository=da,
                             last_change_positions=db, gap_days=gap, depository=qa, positions=qb,
                             diff=qa - qb, mismatch_type="STALE_DEPOSITORY" if da < db
                             else "STALE_POSITIONS"))
    out = frame(rows, ["customer_id", "canonical", "last_change_depository",
                       "last_change_positions", "gap_days", "depository", "positions", "diff",
                       "mismatch_type"])
    return CheckResult(spec, out, denominator=len(set(dep) | set(pos)))


@check(id="B8", title="Calculation version drift", family="B", scope="within_chain2",
       severity="medium", owner="Engineering (ledger)",
       question="Does any position series mix calculation versions?")
def b8_version_drift(ds) -> CheckResult:
    spec = b8_version_drift.spec
    d = dedupe_daily(ds)
    v = d.groupby(["customer_id", "symbol"])["calc_version"].agg(["nunique", "min", "max"])
    bad = v[v["nunique"] > 1].reset_index()
    bad["diff"] = bad["max"] - bad["min"]
    bad["mismatch_type"] = "VERSION_DRIFT"
    out = bad.rename(columns={"nunique": "versions"})[
        ["customer_id", "symbol", "versions", "min", "max", "diff", "mismatch_type"]]
    return CheckResult(spec, out, denominator=len(v),
                       notes=["a series that mixes versions is not internally comparable"])
