"""Family A: internal consistency. Does the data agree with itself?

Gates (A9, A10, A11) run first because they decide whether joins in later checks are sound.
Within-chain checks find derivation defects (assign to engineering). Cross-chain checks are
reconciliation findings and are labelled as such.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .. import model as m
from ..pipeline import latest_depository
from .base import CheckResult, check, frame
from .common import (Series, chain1_rows, chain2_rows, dedupe_daily, equal, lag_explained,
                     live_events, ordered, unit_series, with_canonical)

KEY = ["customer_id", "symbol"]


def _calculated(ds) -> pd.DataFrame:
    ev = live_events(ds)
    return ev[ev["calc_status"] == m.CALCULATED]


# ======================================================================================
# Gates
# ======================================================================================
@check(id="A11", title="Symbol fragmentation and ISIN hygiene", family="A", scope="gate",
       severity="critical", owner="Engineering (reference data)",
       question="Can one security be reached through more than one symbol key?",
       tolerance="exact; check-digit variants are counted, not reported")
def a11_fragmentation(ds) -> CheckResult:
    spec = a11_fragmentation.spec
    sym = ds["symbols"].copy()
    sym["canonical"] = sym["isin"].map(m.canonical_isin)
    groups = sym.groupby("canonical")["symbol"].apply(lambda s: sorted(set(s)))
    fragmented = groups[groups.map(len) > 1]
    holdings = pd.concat([ds[t][KEY] for t in ("depository_daily", "depository_latest",
                                              "positions_daily", "positions_latest")])
    holdings = holdings.drop_duplicates()
    rows, exposed = [], 0
    for canon, symbols in fragmented.items():
        held = holdings[holdings["symbol"].isin(symbols)]
        by_cust = held.groupby("customer_id")["symbol"].apply(lambda s: sorted(set(s)))
        exposed += len(by_cust)
        rows.append(dict(customer_id=None, canonical=canon, symbols=",".join(symbols),
                         symbols_used="", mismatch_type="FRAGMENTED_SECURITY"))
        for cust, used in by_cust.items():
            if len(used) > 1:     # one position, stored under two keys in different tables
                rows.append(dict(customer_id=cust, canonical=canon, symbols=",".join(symbols),
                                 symbols_used=",".join(used),
                                 mismatch_type="SPLIT_ACROSS_SYMBOLS"))
    # (iii) ISIN hygiene across every table that carries an ISIN
    obs = []
    sources = [("symbols", ds["symbols"]["isin"]), ("depository_feed", ds["depository_feed"]["isin"]),
               ("depository_daily", ds["depository_daily"]["received_isin"])]
    for table, col in sources:
        for isin in col.dropna().unique():
            for issue in m.isin_issues(isin):
                kind = "KNOWN_VARIANT" if issue == "CHECK_DIGIT" else "MALFORMED_ISIN"
                if kind == "MALFORMED_ISIN":
                    rows.append(dict(customer_id=None, canonical=m.canonical_isin(isin),
                                     symbols=table, symbols_used=isin, mismatch_type=kind))
                else:
                    obs.append(dict(table=table, isin=isin, issue=issue))
    cols = ["customer_id", "canonical", "symbols", "symbols_used", "mismatch_type"]
    return CheckResult(spec, frame(rows, cols), denominator=int(sym["canonical"].nunique()),
                       observations=pd.DataFrame(obs),
                       metrics=dict(fragmented_securities=len(fragmented), customers_exposed=exposed,
                                    check_digit_variants=len(obs)),
                       notes=["key used: canonical ISIN (first 11 characters)"])


@check(id="A9", title="Duplicate keys", family="A", scope="gate", severity="high",
       owner="Engineering (storage)", question="Is every table unique on its expected grain?")
def a9_duplicates(ds) -> CheckResult:
    spec = a9_duplicates.spec
    grains = {
        "positions_daily": ["customer_id", "symbol", "date"],
        "positions_latest": KEY,
        "depository_daily": ["customer_id", "symbol", "date"],
        "depository_latest": KEY,
        "prices": ["symbol", "date"],
        "symbols": ["symbol"],
    }
    rows, denom = [], 0
    for table, grain in grains.items():
        df = ds[table]
        denom += len(df)
        dup = df[df.duplicated(grain, keep=False)]
        for key, g in dup.groupby(grain, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            exact = len(g.drop_duplicates()) == 1
            rows.append(dict(table=table, customer_id=dict(zip(grain, key)).get("customer_id"),
                             key="|".join(str(k)[:10] for k in key), copies=len(g),
                             mismatch_type="EXACT_DUPLICATE" if exact else "CONFLICTING_DUPLICATE"))
    ev = live_events(ds)
    grain = ["customer_id", "symbol", "date", "time", "event_type"]
    for key, g in ev[ev.duplicated(grain, keep=False)].groupby(grain):
        rows.append(dict(table="position_events", customer_id=key[0],
                         key="|".join(str(k)[:10] for k in key), copies=len(g),
                         mismatch_type="DUPLICATE_EVENT"))
    sym = ds["symbols"]
    for isin, g in sym[sym.duplicated("isin", keep=False)].groupby("isin"):
        rows.append(dict(table="symbols", customer_id=None, key=isin, copies=len(g),
                         mismatch_type="DUPLICATE_ISIN"))
    return CheckResult(spec, frame(rows, ["table", "customer_id", "key", "copies", "mismatch_type"]),
                       denominator=denom + len(ev))


@check(id="A10", title="Orphaned references", family="A", scope="gate", severity="high",
       owner="Engineering (storage)",
       question="Does every customer, symbol and corporate-action reference resolve?")
def a10_orphans(ds) -> CheckResult:
    spec = a10_orphans.spec
    symbols = set(ds["symbols"]["symbol"])
    customers = set(ds["customers"]["customer_id"])
    cas = set(ds["corporate_actions"]["ca_id"])
    rows, denom = [], 0
    for table in ("depository_daily", "depository_latest", "position_events", "positions_daily",
                  "positions_latest", "app_display"):
        df = ds[table]
        denom += len(df)
        for r in df[~df["symbol"].isin(symbols)].itertuples(index=False):
            rows.append(dict(table=table, customer_id=r.customer_id, reference=r.symbol,
                             mismatch_type="UNKNOWN_SYMBOL"))
        for r in df[~df["customer_id"].isin(customers)].itertuples(index=False):
            rows.append(dict(table=table, customer_id=r.customer_id, reference=r.customer_id,
                             mismatch_type="UNKNOWN_CUSTOMER"))
    for table in ("prices", "corporate_actions"):
        df = ds[table]
        for s in sorted(set(df["symbol"]) - symbols):
            rows.append(dict(table=table, customer_id=None, reference=s,
                             mismatch_type="UNKNOWN_SYMBOL"))
    ev = ds["position_events"]
    bad = ev[ev["ca_id"].notna() & ~ev["ca_id"].isin(cas)]
    for r in bad.itertuples(index=False):
        rows.append(dict(table="position_events", customer_id=r.customer_id, reference=r.ca_id,
                         mismatch_type="UNKNOWN_CORPORATE_ACTION"))
    return CheckResult(spec, frame(rows, ["table", "customer_id", "reference", "mismatch_type"]),
                       denominator=denom)


# ======================================================================================
# Within-chain: depository route
# ======================================================================================
@check(id="A2W", title="Depository snapshot equals its latest history row", family="A",
       scope="within_chain1", severity="high", owner="Engineering (depository intake)",
       question="Is depository_latest exactly the last non-ignored depository_daily row?")
def a2w_depository_snapshot(ds) -> CheckResult:
    spec = a2w_depository_snapshot.spec
    expected = latest_depository(ds["depository_daily"])
    stored = ds["depository_latest"]
    mg = stored.merge(expected, on=KEY, how="outer", suffixes=("_snapshot", "_history"),
                      indicator="side")
    rows = []
    for r in mg.itertuples(index=False):
        base = dict(customer_id=r.customer_id, symbol=r.symbol, snapshot=r.quantity_snapshot,
                    history=r.quantity_history, date_snapshot=r.date_snapshot,
                    date_history=r.date_history)
        if r.side == "left_only":
            rows.append(dict(base, diff=r.quantity_snapshot, mismatch_type="MISSING_IN_HISTORY"))
        elif r.side == "right_only":
            rows.append(dict(base, diff=-r.quantity_history, mismatch_type="MISSING_IN_SNAPSHOT"))
        elif not equal(r.quantity_snapshot, r.quantity_history):
            rows.append(dict(base, diff=r.quantity_snapshot - r.quantity_history,
                             mismatch_type="ASSET_MISMATCH"))
        elif r.date_snapshot != r.date_history:
            rows.append(dict(base, diff=0, mismatch_type="DATE_MISMATCH"))
    cols = ["customer_id", "symbol", "snapshot", "history", "diff", "date_snapshot",
            "date_history", "mismatch_type"]
    out = frame(rows, cols)
    if len(out):
        out = out.reindex(out["diff"].abs().sort_values(ascending=False).index)
    return CheckResult(spec, out, denominator=len(mg))


@check(id="A8", title="Received ISIN identity", family="A", scope="within_chain1",
       severity="medium", owner="Engineering (depository intake)",
       question="Can every received ISIN be tied to the symbol it was stored under?",
       tolerance="12th-character variants and linked symbol changes are explained")
def a8_isin_identity(ds) -> CheckResult:
    spec = a8_isin_identity.spec
    isin_of = dict(zip(ds["symbols"]["symbol"], ds["symbols"]["isin"]))
    lin = ds["isin_lineage"]
    link = {m.canonical_isin(o): m.canonical_isin(n) for o, n in zip(lin["old_isin"], lin["new_isin"])}

    def resolves(c_from, c_to):
        for _ in range(5):
            if c_from == c_to:
                return True
            if c_from not in link:
                return False
            c_from = link[c_from]
        return False

    rows, counts = [], defaultdict(int)
    dd = ds["depository_daily"]
    for r in dd.itertuples(index=False):
        own = isin_of.get(r.symbol)
        if own is None or not isinstance(r.received_isin, str):
            continue
        if r.received_isin == own:
            counts["EXACT"] += 1
        elif m.canonical_isin(r.received_isin) == m.canonical_isin(own):
            counts["VARIANT"] += 1
        elif resolves(m.canonical_isin(r.received_isin), m.canonical_isin(own)):
            counts["LINKED_CHANGE"] += 1
        else:
            counts["UNLINKED"] += 1
            rows.append(dict(customer_id=r.customer_id, symbol=r.symbol, date=r.date,
                             received_isin=r.received_isin, symbol_isin=own,
                             mismatch_type="UNLINKED_ISIN_CHANGE"))
    out = frame(rows, ["customer_id", "symbol", "date", "received_isin", "symbol_isin",
                       "mismatch_type"])
    return CheckResult(spec, out, denominator=len(dd), metrics=dict(counts),
                       notes=["only character 12 differs -> same security (counted, not reported)"])


# ======================================================================================
# Within-chain: transactional route
# ======================================================================================
@check(id="A0", title="Hard invariants (negative holdings, impossible totals)", family="A",
       scope="within_chain2", severity="critical", owner="Engineering (ledger)",
       question="Are there values that are impossible under any business rule?")
def a0_invariants(ds) -> CheckResult:
    spec = a0_invariants.spec
    rows, gradient = [], {}
    ev = _calculated(ds)
    tests = [("balance_after", "NEGATIVE_BALANCE"), ("avg_cost_after", "NEGATIVE_AVG_COST"),
             ("cash_amount", "NEGATIVE_DIVIDEND"), ("quantity", "NEGATIVE_QUANTITY")]
    for col, label in tests:
        for r in ev[ev[col] < 0].itertuples(index=False):
            v = getattr(r, col)
            rows.append(dict(source="position_events", customer_id=r.customer_id,
                             symbol=r.symbol, date=r.date, time=r.time, value=v, diff=v,
                             mismatch_type=label))
    gradient["position_events"] = sum(1 for r in rows if r["source"] == "position_events")
    for table in ("positions_daily", "positions_latest"):
        df = dedupe_daily(ds) if table == "positions_daily" else ds[table]
        acquired = (df["bought_total"] + df["bonus_total"] + df["rights_total"]
                    + df["transfer_in_total"])
        checks = [(df["quantity"] < 0, df["quantity"], "NEGATIVE_BALANCE"),
                  (df["avg_cost"] < 0, df["avg_cost"], "NEGATIVE_AVG_COST"),
                  (df["sold_total"] > acquired, acquired - df["sold_total"],
                   "SOLD_EXCEEDS_ACQUIRED")]
        n = 0
        for mask, value, label in checks:
            for i in df.index[mask]:
                r = df.loc[i]
                rows.append(dict(source=table, customer_id=r["customer_id"], symbol=r["symbol"],
                                 date=r["date"], time=None, value=float(value[i]),
                                 diff=float(value[i]), mismatch_type=label))
                n += 1
        gradient[table] = n
    notes = []
    neg_ledger = {(r["customer_id"], r["symbol"]) for r in rows
                  if r["source"] == "position_events" and r["mismatch_type"] == "NEGATIVE_BALANCE"}
    neg_daily = {(r["customer_id"], r["symbol"]) for r in rows
                 if r["source"] == "positions_daily" and r["mismatch_type"] == "NEGATIVE_BALANCE"}
    if neg_ledger - neg_daily:
        notes.append(f"{len(neg_ledger - neg_daily)} unit(s) negative in the ledger but not in "
                     "the daily table: the rollup is masking them")
    cols = ["source", "customer_id", "symbol", "date", "time", "value", "diff", "mismatch_type"]
    return CheckResult(spec, frame(rows, cols), denominator=len(ev) + len(ds["positions_daily"]),
                       metrics=dict(gradient=gradient), notes=notes)


@check(id="A1", title="Unattributed share movement (gap plugs)", family="A", scope="cross_chain",
       severity="high", owner="Data reconciliation",
       question="How often does the ledger need a plug to agree with the depository?",
       tolerance="exact; increases and decreases reported separately")
def a1_unattributed(ds) -> CheckResult:
    spec = a1_unattributed.spec
    ev = live_events(ds)
    plugs = ev[ev["event_type"].isin(m.GAP_EVENT_TYPES)].copy()
    plugs["diff"] = [m.signed_quantity(t, q) for t, q in zip(plugs["event_type"], plugs["quantity"])]
    plugs["value"] = plugs["diff"] * plugs["price"]
    plugs["mismatch_type"] = plugs["event_type"]
    out = plugs[["customer_id", "symbol", "date", "time", "quantity", "price", "value", "diff",
                 "mismatch_type"]]
    inc = plugs[plugs["event_type"] == m.UNEXPLAINED_INCREASE]
    dec = plugs[plugs["event_type"] == m.UNEXPLAINED_DECREASE]
    metrics = dict(per_1000_events=round(1000 * len(plugs) / max(len(ev), 1), 2),
                   increases=len(inc), decreases=len(dec),
                   increase_value=float(inc["value"].sum()), decrease_value=float(dec["value"].sum()))
    return CheckResult(spec, out.reset_index(drop=True), denominator=len(ev), metrics=metrics,
                       notes=["a plug may hide a real defect or a legitimate manual registration; "
                              "interpret before calling it a defect"])


@check(id="A3", title="Position snapshot equals its latest daily row", family="A",
       scope="within_chain2", severity="high", owner="Engineering (ledger)",
       question="Is positions_latest the last positions_daily row, on every column?",
       tolerance="1e-6 on numeric columns")
def a3_snapshot(ds) -> CheckResult:
    spec = a3_snapshot.spec
    daily = dedupe_daily(ds).sort_values("date")
    last = daily.groupby(KEY).tail(1).set_index(KEY)
    latest = ds["positions_latest"].drop_duplicates(KEY, keep="last").set_index(KEY)
    columns = [c for c in latest.columns]
    rows, agree = [], defaultdict(int)
    both = latest.index.intersection(last.index)
    for key in latest.index.union(last.index):
        if key not in last.index or key not in latest.index:
            rows.append(dict(customer_id=key[0], symbol=key[1], column="*", snapshot=None,
                             history=None, diff=None,
                             mismatch_type="MISSING_IN_HISTORY" if key not in last.index
                             else "MISSING_IN_SNAPSHOT"))
    for key in both:
        a, b = latest.loc[key], last.loc[key]
        for c in columns:
            if equal(a[c], b[c], 1e-6):
                agree[c] += 1
            else:
                diff = None
                try:
                    diff = float(a[c]) - float(b[c])
                except (TypeError, ValueError):
                    pass
                rows.append(dict(customer_id=key[0], symbol=key[1], column=c, snapshot=a[c],
                                 history=b[c], diff=diff,
                                 mismatch_type="DATE_MISMATCH" if c == "date" else "STALE_VALUE"))
    rates = {c: round(agree[c] / len(both), 4) if len(both) else None for c in columns}
    out = frame(rows, ["customer_id", "symbol", "column", "snapshot", "history", "diff",
                       "mismatch_type"])
    return CheckResult(spec, out, denominator=len(both), metrics=dict(column_agreement=rates),
                       notes=["snapshot lineage assumed (latest = last daily row); confirm with "
                              "the data owner"])


@check(id="A4", title="Ledger-to-daily rollup", family="A", scope="within_chain2",
       severity="critical", owner="Engineering (ledger)",
       question="Do daily positions equal the ledger? (i) signed rollup, (ii) terminal balance",
       tolerance="exact")
def a4_rollup(ds) -> CheckResult:
    spec = a4_rollup.spec
    ev = ordered(_calculated(ds))
    daily = dedupe_daily(ds).sort_values("date")
    dq = {(r.customer_id, r.symbol, r.date): r.quantity for r in daily.itertuples(index=False)}
    prev_q = {}
    for (c, s), g in daily.groupby(KEY):
        q_prev = 0
        for r in g.itertuples(index=False):
            prev_q[(c, s, r.date)] = q_prev
            q_prev = r.quantity
    rows, residuals = [], []
    for (c, s, d), g in ev.groupby(["customer_id", "symbol", "date"], sort=False):
        key = (c, s, d)
        terminal = g["balance_after"].iloc[-1]
        flow = sum(m.signed_quantity(t, q) for t, q in zip(g["event_type"], g["quantity"]))
        if key not in dq:
            rows.append(dict(assertion="terminal", customer_id=c, symbol=s, date=d,
                             ledger=terminal, daily=None, diff=terminal,
                             mismatch_type="MISSING_DAILY_ROW"))
            continue
        if not equal(terminal, dq[key]):
            rows.append(dict(assertion="terminal", customer_id=c, symbol=s, date=d,
                             ledger=terminal, daily=dq[key], diff=dq[key] - terminal,
                             mismatch_type="TERMINAL_BALANCE_BREAK"))
        change = dq[key] - prev_q[key]
        residuals.append(change - flow)
        if not equal(change, flow):
            rows.append(dict(assertion="rollup", customer_id=c, symbol=s, date=d, ledger=flow,
                             daily=change, diff=change - flow, mismatch_type="ROLLUP_BREAK"))
    ledger_days = set(zip(ev["customer_id"], ev["symbol"], ev["date"]))
    for key in set(dq) - ledger_days:
        rows.append(dict(assertion="terminal", customer_id=key[0], symbol=key[1], date=key[2],
                         ledger=None, daily=dq[key], diff=dq[key],
                         mismatch_type="DAILY_ROW_WITHOUT_EVENTS"))
    res = pd.Series(residuals, dtype=float)
    nz = res[res != 0]
    metrics = dict(days_checked=len(res), residual_nonzero=len(nz),
                   residual_abs_p50=float(nz.abs().median()) if len(nz) else 0.0,
                   residual_abs_max=float(nz.abs().max()) if len(nz) else 0.0)
    out = frame(rows, ["assertion", "customer_id", "symbol", "date", "ledger", "daily", "diff",
                       "mismatch_type"])
    return CheckResult(spec, out, denominator=len(ledger_days), metrics=metrics)


@check(id="A5B", title="Running balance continuity in the ledger", family="A",
       scope="within_chain2", severity="critical", owner="Engineering (ledger)",
       question="Does balance_after[t] = balance_after[t-1] + signed quantity[t]?")
def a5b_continuity(ds) -> CheckResult:
    spec = a5b_continuity.spec
    ev = ordered(live_events(ds))
    rows, checked, by_type = [], 0, defaultdict(int)
    for (c, s), g in ev.groupby(KEY, sort=False):
        prev = 0.0
        for r in g.itertuples(index=False):
            if r.calc_status != m.CALCULATED or pd.isna(r.balance_after):
                prev = None                  # uncalculated row: continuity resumes after it
                continue
            if prev is not None:
                checked += 1
                expected = prev + m.signed_quantity(r.event_type, r.quantity)
                if not equal(r.balance_after, expected):
                    by_type[r.event_type] += 1
                    rows.append(dict(customer_id=c, symbol=s, date=r.date, time=r.time,
                                     event_type=r.event_type, expected=expected,
                                     stored=r.balance_after, diff=r.balance_after - expected,
                                     mismatch_type="BALANCE_BREAK"))
            prev = r.balance_after
    out = frame(rows, ["customer_id", "symbol", "date", "time", "event_type", "expected",
                       "stored", "diff", "mismatch_type"])
    return CheckResult(spec, out, denominator=checked, metrics=dict(breaks_by_event=dict(by_type)),
                       notes=["breaks on plug rows corroborate A1; breaks on trades are worse"])


@check(id="A6", title="Average cost changes only on paid inflows", family="A",
       scope="within_chain2", severity="medium", owner="Engineering (ledger)",
       question="Does any sale or zero-quantity event change the average cost?",
       tolerance="1e-4 IRR")
def a6_avg_cost(ds) -> CheckResult:
    spec = a6_avg_cost.spec
    ev = ordered(_calculated(ds))
    rows, checked = [], 0
    for (c, s), g in ev.groupby(KEY, sort=False):
        prev = 0.0
        for r in g.itertuples(index=False):
            if pd.isna(r.avg_cost_after):
                continue
            if m.DIRECTION[r.event_type] <= 0 and r.balance_after > 0:
                checked += 1
                if not equal(r.avg_cost_after, prev, 1e-4):
                    rows.append(dict(customer_id=c, symbol=s, date=r.date, time=r.time,
                                     event_type=r.event_type, before=prev,
                                     after=r.avg_cost_after, diff=r.avg_cost_after - prev,
                                     mismatch_type="AVG_COST_CHANGED_ON_OUTFLOW"))
            prev = r.avg_cost_after
    out = frame(rows, ["customer_id", "symbol", "date", "time", "event_type", "before", "after",
                       "diff", "mismatch_type"])
    return CheckResult(spec, out, denominator=checked)


@check(id="A12", title="Corporate-action pass-through", family="A", scope="within_chain2",
       severity="critical", owner="Engineering (corporate actions)",
       question="Did every holder on the record date receive the corporate-action row?",
       tolerance="exact; each action type reported separately")
def a12_ca_passthrough(ds) -> CheckResult:
    spec = a12_ca_passthrough.spec
    cal = ds.calendar
    daily = dedupe_daily(ds)
    ev = live_events(ds)
    expected_type = {m.DIVIDEND: m.CASH_DIVIDEND, m.BONUS: m.BONUS_GRANT, m.RIGHTS: m.RIGHTS_GRANT}
    rows, per_action, denom = [], {}, 0
    for ca in ds["corporate_actions"].itertuples(index=False):
        if ca.action_type in (m.DIVIDEND, m.BONUS) and not ca.record_date > ca.session_date:
            rows.append(dict(ca_id=ca.ca_id, action_type=ca.action_type, customer_id=None,
                             expected=None, received=None, diff=None, mismatch_type="CA_DATE_RULE"))
        if ca.action_type == m.RIGHTS and ca.record_date != ca.session_date:
            rows.append(dict(ca_id=ca.ca_id, action_type=ca.action_type, customer_id=None,
                             expected=None, received=None, diff=None, mismatch_type="CA_DATE_RULE"))
        if ca.action_type == m.BONUS and not ca.value > 0:
            rows.append(dict(ca_id=ca.ca_id, action_type=ca.action_type, customer_id=None,
                             expected=None, received=None, diff=None, mismatch_type="CA_VALUE_RULE"))
        before = cal.shift(ca.record_date, -1)
        d = daily[(daily["symbol"] == ca.symbol) & (daily["date"] <= before)].sort_values("date")
        held = d.groupby("customer_id")["quantity"].last()
        held = held[held > 0]
        etype = expected_type[ca.action_type]
        got = ev[(ev["ca_id"] == ca.ca_id) & (ev["event_type"] == etype)]
        got = got.groupby("customer_id").agg(quantity=("quantity", "sum"),
                                             cash=("cash_amount", "sum"))
        eligible = 0
        for cust, q in held.items():
            if ca.action_type == m.DIVIDEND:
                exp = round(q * ca.value, 2)
                rec = got["cash"].get(cust)
            else:
                exp = m.entitlement(q, ca.value)
                if exp == 0:
                    continue
                rec = got["quantity"].get(cust)
            eligible += 1
            if rec is None or pd.isna(rec):
                rows.append(dict(ca_id=ca.ca_id, action_type=ca.action_type, customer_id=cust,
                                 expected=exp, received=0, diff=-exp,
                                 mismatch_type="MISSING_CA_EVENT"))
            elif not equal(rec, exp, 0.01):
                rows.append(dict(ca_id=ca.ca_id, action_type=ca.action_type, customer_id=cust,
                                 expected=exp, received=rec, diff=rec - exp,
                                 mismatch_type="AMOUNT_MISMATCH"))
        for cust in sorted(set(got.index) - set(held.index)):
            rows.append(dict(ca_id=ca.ca_id, action_type=ca.action_type, customer_id=cust,
                             expected=0, received=got["quantity"].get(cust), diff=None,
                             mismatch_type="UNEXPECTED_CA_EVENT"))
        denom += eligible
        per_action[ca.ca_id] = dict(type=ca.action_type, eligible=eligible,
                                    received=int(got.index.isin(held.index).sum()))
    out = frame(rows, ["ca_id", "action_type", "customer_id", "expected", "received", "diff",
                       "mismatch_type"])
    return CheckResult(spec, out, denominator=denom, metrics=dict(per_action=per_action),
                       notes=["dividend gaps are a direct money-leak risk; bonus and rights gaps "
                              "are share-count errors"])


# ======================================================================================
# Cross-chain: depository route vs transactional route
# ======================================================================================
def _lag_tolerant(dep: Series, pos: Series, d, cal, lag_days) -> bool:
    a, b = dep.at(d), pos.at(d)
    return lag_explained(dep, b, d, cal, lag_days) or lag_explained(pos, a, d, cal, lag_days)


def _recent_lag(moving: Series, other_q: float, cal, lag_days: int) -> bool:
    """True if ``moving`` changed within ``lag_days`` of the as-of date and its value just
    before that change equals the other side's value (the other side has not caught up yet)."""
    last = moving.change_dates()[-1:]
    if not last or cal.between(last[0], cal.as_of) > lag_days:
        return False
    return moving.at(cal.shift(last[0], -1)) == other_q


@check(id="A2X_LATEST", title="Depository snapshot vs position snapshot", family="A",
       scope="cross_chain", severity="high", owner="Data reconciliation",
       question="Do the two routes agree on today's share count?",
       tolerance="exact on the canonical ISIN key; settlement lag up to 2 trading days")
def a2x_latest(ds, lag_days: int = 2) -> CheckResult:
    spec = a2x_latest.spec
    dep = with_canonical(ds, ds["depository_latest"])
    pos = with_canonical(ds, ds["positions_latest"].drop_duplicates(KEY, keep="last"))
    k = ["customer_id", "canonical"]
    a = dep.groupby(k).agg(depository=("quantity", "sum"), date_depository=("date", "max"))
    b = pos.groupby(k).agg(positions=("quantity", "sum"), date_positions=("date", "max"))
    mg = a.join(b, how="outer")
    dep_s, pos_s = unit_series(chain1_rows(ds)), unit_series(chain2_rows(ds))
    cal = ds.calendar
    rows, obs = [], []
    for (c, canon), r in mg.iterrows():
        qa = 0.0 if pd.isna(r["depository"]) else r["depository"]
        qb = 0.0 if pd.isna(r["positions"]) else r["positions"]
        if qa == qb:
            continue
        rec = dict(customer_id=c, canonical=canon, depository=r["depository"],
                   positions=r["positions"], diff=qa - qb, date_depository=r["date_depository"],
                   date_positions=r["date_positions"])
        ds_, ps_ = dep_s.get((c, canon), Series([])), pos_s.get((c, canon), Series([]))
        if _recent_lag(ps_, qa, cal, lag_days) or _recent_lag(ds_, qb, cal, lag_days):
            obs.append(dict(rec, mismatch_type="SETTLEMENT_LAG"))
            continue
        kind = ("MISSING_IN_DEPOSITORY" if pd.isna(r["depository"]) else
                "MISSING_IN_POSITIONS" if pd.isna(r["positions"]) else "ASSET_MISMATCH")
        rows.append(dict(rec, mismatch_type=kind))
    cols = ["customer_id", "canonical", "depository", "positions", "diff", "date_depository",
            "date_positions", "mismatch_type"]
    out = frame(rows, cols)
    if len(out):
        out = out.reindex(out["diff"].abs().sort_values(ascending=False).index)
    return CheckResult(spec, out, denominator=len(mg), observations=pd.DataFrame(obs),
                       notes=["key used: customer + canonical ISIN"])


@check(id="A2X_HIST", title="Depository history vs daily positions", family="A",
       scope="cross_chain", severity="medium", owner="Data reconciliation",
       question="Do the two routes agree on every day either of them changed?",
       tolerance="exact; differences that close within 2 trading days are settlement lag")
def a2x_history(ds, lag_days: int = 2) -> CheckResult:
    spec = a2x_history.spec
    cal = ds.calendar
    dep, pos = unit_series(chain1_rows(ds)), unit_series(chain2_rows(ds))
    rows, lagged, checked = [], 0, 0
    for unit in sorted(set(dep) | set(pos), key=str):
        a, b = dep.get(unit, Series([])), pos.get(unit, Series([]))
        for d in sorted(set(a.change_dates()) | set(b.change_dates())):
            checked += 1
            qa, qb = a.at(d), b.at(d)
            if qa == qb:
                continue
            if _lag_tolerant(a, b, d, cal, lag_days):
                lagged += 1
                continue
            rows.append(dict(customer_id=unit[0], canonical=unit[1], date=d, depository=qa,
                             positions=qb, diff=qa - qb, mismatch_type="ASSET_MISMATCH"))
    out = frame(rows, ["customer_id", "canonical", "date", "depository", "positions", "diff",
                       "mismatch_type"])
    return CheckResult(spec, out, denominator=checked, metrics=dict(settlement_lag_days=lagged),
                       notes=["key used: customer + canonical ISIN"])
