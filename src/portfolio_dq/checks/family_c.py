"""Family C: does the customer see the right number?"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .. import model as m
from ..pipeline import HORIZONS, build_app_display, portfolio_values
from .base import CheckResult, check, frame
from .common import dedupe_daily, equal

DISPLAY_FIELDS = ["quantity", "market_value", "cost_value", "unrealized_pnl", "return_pct",
                  "break_even_price"]


@check(id="C1", title="App display recomputation", family="C", scope="cross_system",
       severity="critical", owner="App",
       question="Recompute the portfolio screen from stored positions: does the app agree?",
       tolerance="1 IRR on amounts, 1e-6 on return")
def c1_display(ds) -> CheckResult:
    """market value = shares x close x (1 - sell fee); cost = shares x average cost;
    unrealized P&L = market value - cost; return = P&L / cost;
    break-even = average cost x (1 + sell fee). The sell fee depends on instrument class."""
    spec = c1_display.spec
    latest = ds["positions_latest"].drop_duplicates(["customer_id", "symbol"], keep="last")
    expected = build_app_display(latest, ds["symbols"], ds["fee_schedule"], ds["prices"], ds.as_of)
    app = ds["app_display"]
    key = ["customer_id", "symbol"]
    mg = app.merge(expected, on=key, how="outer", suffixes=("_app", "_calc"), indicator="side")
    rows = []
    for _, r in mg.iterrows():
        if r["side"] != "both":
            rows.append(dict(customer_id=r["customer_id"], symbol=r["symbol"], field="*",
                             app=None, recomputed=None, diff=None,
                             mismatch_type="MISSING_IN_APP" if r["side"] == "right_only"
                             else "MISSING_IN_POSITIONS"))
            continue
        for f in DISPLAY_FIELDS:
            tol = 1e-6 if f == "return_pct" else 1.0
            a, b = r[f + "_app"], r[f + "_calc"]
            if not equal(a, b, tol):
                diff = None if pd.isna(a) or pd.isna(b) else float(a) - float(b)
                rows.append(dict(customer_id=r["customer_id"], symbol=r["symbol"], field=f,
                                 app=a, recomputed=b, diff=diff, mismatch_type="VALUE_MISMATCH"))
    out = frame(rows, ["customer_id", "symbol", "field", "app", "recomputed", "diff",
                       "mismatch_type"])
    return CheckResult(spec, out, denominator=len(mg),
                       notes=["the share count comes from the transactional route"])


@check(id="C3", title="Return engine values vs holdings", family="C", scope="cross_system",
       severity="high", owner="Return engine",
       question="Do the start and end values used for returns equal the value of the holdings?",
       tolerance="max(1 IRR, 0.01%)")
def c3_return_values(ds) -> CheckResult:
    spec = c3_return_values.spec
    cal, daily, prices = ds.calendar, dedupe_daily(ds), ds["prices"]
    days = dict(HORIZONS)
    cache = {}

    def value(cust, day):
        if day not in cache:
            cache[day] = portfolio_values(daily, prices, day)["value"]
        return float(cache[day].get(cust, 0.0))

    out_rows, n = [], 0
    ro = ds["return_outputs"]
    for r in ro[ro["status"] == "OK"].itertuples(index=False):
        if r.horizon not in days:
            continue
        start_day = cal.shift(r.as_of, -days[r.horizon])
        for side, stored, day in (("start", r.start_value, start_day), ("end", r.end_value, r.as_of)):
            n += 1
            calc = value(r.customer_id, day)
            if not equal(stored, calc, max(1.0, abs(calc) * 1e-4)):
                out_rows.append(dict(customer_id=r.customer_id, horizon=r.horizon, side=side,
                                     return_engine=stored, holdings=round(calc), diff=stored - calc,
                                     mismatch_type="VALUE_MISMATCH"))
    out = frame(out_rows, ["customer_id", "horizon", "side", "return_engine", "holdings", "diff",
                           "mismatch_type"])
    return CheckResult(spec, out, denominator=n)


@check(id="C4", title="Return status coverage", family="C", scope="cross_system",
       severity="medium", owner="Return engine",
       question="Is every return status a known code, and what share is displayable?")
def c4_status(ds) -> CheckResult:
    spec = c4_status.spec
    ro = ds["return_outputs"]
    bad = ro[~ro["status"].isin(m.RETURN_STATUSES)]
    out = pd.DataFrame(dict(customer_id=bad["customer_id"], horizon=bad["horizon"],
                            status=bad["status"], mismatch_type="UNKNOWN_STATUS"))
    coverage = {h: dict(Counter(g["status"])) for h, g in ro.groupby("horizon")}
    ok = {h: round(float((g["status"] == "OK").mean()), 4) for h, g in ro.groupby("horizon")}
    return CheckResult(spec, out.reset_index(drop=True), denominator=len(ro),
                       metrics=dict(status_counts=coverage, ok_share=ok),
                       notes=["an unknown code is surfaced, never bucketed into 'other'"])


@check(id="C5", title="External cash-flow completeness", family="C", scope="cross_system",
       severity="high", owner="Return engine",
       question="Does the return engine use exactly the deposits and withdrawals, and nothing else?",
       tolerance="exact on customer, date and amount")
def c5_cash_flows(ds) -> CheckResult:
    spec = c5_cash_flows.spec
    cash = ds["cash_movements"]
    ext = cash[cash["kind"].isin(m.EXTERNAL_FLOW_KINDS)]
    key = lambda df: Counter(zip(df["customer_id"], df["date"], df["amount"].round(2)))  # noqa: E731
    want, got = key(ext), key(ds["return_cash_flows"])
    internal = {(c, d, round(a, 2)): k for c, d, a, k in
                zip(cash["customer_id"], cash["date"], cash["amount"], cash["kind"])
                if k not in m.EXTERNAL_FLOW_KINDS}
    rows = []
    for k, n in (got - want).items():
        rows += [dict(customer_id=k[0], date=k[1], amount=k[2], diff=k[2],
                      matches=internal.get(k, ""), mismatch_type="NOT_AN_EXTERNAL_FLOW")] * n
    for k, n in (want - got).items():
        rows += [dict(customer_id=k[0], date=k[1], amount=k[2], diff=-k[2], matches="",
                      mismatch_type="MISSING_FLOW")] * n
    out = frame(rows, ["customer_id", "date", "amount", "diff", "matches", "mismatch_type"])
    return CheckResult(spec, out, denominator=len(ext),
                       notes=["dividends, rights payments and credit are internal flows"])
