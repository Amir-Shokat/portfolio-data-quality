"""Reference derivations for both holdings routes.

These functions are the "correct pipeline" of the synthetic world. The generator uses them
to produce clean stored data, and the replay engine uses them to rebuild a customer in
memory and compare with what is stored.

Depository route (share counts only)::

    depository_feed  ->  depository_daily  ->  depository_latest

Transactional route (full economics)::

    source events + corporate actions + depository_daily (gap plugs)
        ->  position_events  ->  positions_daily  ->  positions_latest

Note the dependency: the transactional route reads the depository route to plug gaps, so
the two routes are two derivations that must agree, not two independent sources of truth.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import model as m

CA_TIME = "08:15:00"        # corporate-action rows, before the market opens
ACTIVATION_TIME = "08:30:00"
EXPIRY_TIME = "16:00:00"    # rights lapse after the session
PLUG_TIME = "23:59:00"      # gap plugs, after the depository file for the day
HORIZONS = (("1M", 21), ("2M", 42))   # return windows: label, trading days back from as-of

EVENT_COLUMNS = ["event_id", "customer_id", "symbol", "date", "time", "event_type", "origin",
                 "source_ref", "quantity", "price", "balance_after", "avg_cost_after", "ca_id",
                 "ca_value", "cash_amount", "is_deleted", "calc_status"]
DAILY_COLUMNS = ["customer_id", "symbol", "date", "quantity", "bought_total", "sold_total",
                 "bonus_total", "rights_total", "transfer_in_total", "transfer_out_total",
                 "avg_cost", "realized_pnl", "calc_version"]
CHAIN1_COLUMNS = ["customer_id", "symbol", "date", "quantity", "received_isin", "row_type",
                  "registration_type"]


# ======================================================================================
# Depository route
# ======================================================================================
class IsinResolver:
    """Map a received ISIN to a symbol: exact match, then canonical key, then lineage."""

    def __init__(self, symbols: pd.DataFrame, lineage: pd.DataFrame | None = None):
        self.exact = dict(zip(symbols["isin"], symbols["symbol"]))
        self.by_canonical: dict[str, str] = {}
        for sym, isin in sorted(zip(symbols["symbol"], symbols["isin"])):
            self.by_canonical.setdefault(m.canonical_isin(isin), sym)
        self.lineage = {}
        if lineage is not None and len(lineage):
            self.lineage = {m.canonical_isin(o): n for o, n in zip(lineage["old_isin"],
                                                                   lineage["new_isin"])}

    def resolve(self, isin, _depth: int = 0) -> str | None:
        if isin in self.exact:
            return self.exact[isin]
        c = m.canonical_isin(isin)
        if c in self.by_canonical:
            return self.by_canonical[c]
        if c in self.lineage and _depth < 5:
            return self.resolve(self.lineage[c], _depth + 1)
        return None


@dataclass
class Chain1:
    daily: pd.DataFrame
    latest: pd.DataFrame
    rejected: pd.DataFrame


def build_chain1(feed: pd.DataFrame, symbols: pd.DataFrame,
                 lineage: pd.DataFrame | None = None) -> Chain1:
    """Derive the dated depository history and its latest snapshot from the raw feed.

    Rules:
    * ISINs resolve on the canonical key (first 11 characters), then through ``lineage``.
    * Several ISIN variants for one symbol on one day (a same-day market or ISIN change):
      keep the row with the largest quantity and mark it ``CHANGE_MARKET``.
    * A ``LEAVE`` row (custodian change to zero) directly followed by a ``JOIN`` with the
      same quantity as before the leave is a spurious exit: mark it ``IGNORE``.
    * ``ADMIN`` rows (manual registrations) cannot be rebuilt from settlements; they pass
      through with registration type ``ADMIN``.
    """
    resolver = IsinResolver(symbols, lineage)
    f = feed.copy()
    f["symbol"] = f["isin"].map(resolver.resolve)
    rejected = f[f["symbol"].isna()].copy()
    f = f[f["symbol"].notna()].sort_values(["customer_id", "symbol", "date", "feed_id"])
    rows = []
    for (cust, sym), g in f.groupby(["customer_id", "symbol"], sort=True):
        unit = []
        for day, gd in g.groupby("date", sort=True):
            if gd["isin"].nunique() > 1:
                keep = gd.loc[gd["quantity"].idxmax()]
                row_type = m.ROW_CHANGE_MARKET
            else:
                keep = gd.iloc[-1]
                if keep["origin"] == m.FEED_CUSTODIAN:
                    row_type = m.ROW_LEAVE if keep["quantity"] == 0 else m.ROW_JOIN
                else:
                    row_type = m.ROW_POSITION
            unit.append(dict(customer_id=cust, symbol=sym, date=day, quantity=int(keep["quantity"]),
                             received_isin=keep["isin"], row_type=row_type,
                             registration_type=m.REG_ADMIN if keep["origin"] == m.FEED_ADMIN
                             else m.REG_DEPOSITORY))
        for i in range(1, len(unit) - 1):
            if unit[i]["row_type"] == m.ROW_LEAVE and unit[i + 1]["row_type"] == m.ROW_JOIN:
                before = [r for r in unit[:i] if r["row_type"] != m.ROW_IGNORE]
                if before and before[-1]["quantity"] == unit[i + 1]["quantity"]:
                    unit[i]["row_type"] = m.ROW_IGNORE
        rows.extend(unit)
    daily = pd.DataFrame(rows, columns=CHAIN1_COLUMNS)
    latest = latest_depository(daily)
    return Chain1(daily=daily, latest=latest, rejected=rejected)


def latest_depository(daily: pd.DataFrame) -> pd.DataFrame:
    """Latest non-ignored row per customer and symbol."""
    live = daily[daily["row_type"] != m.ROW_IGNORE].sort_values("date")
    latest = live.groupby(["customer_id", "symbol"], sort=True).tail(1)
    return latest[["customer_id", "symbol", "date", "quantity", "row_type"]].reset_index(drop=True)


# ======================================================================================
# Transactional route
# ======================================================================================
@dataclass
class _Unit:
    balance: int = 0
    avg: float = 0.0
    realized: float = 0.0
    totals: dict = field(default_factory=lambda: dict(bought_total=0, sold_total=0, bonus_total=0,
                                                      rights_total=0, transfer_in_total=0,
                                                      transfer_out_total=0))
    pending_bonus: dict = field(default_factory=dict)
    eod: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    daily: list = field(default_factory=list)


@dataclass
class Chain2:
    events: pd.DataFrame
    daily: pd.DataFrame
    latest: pd.DataFrame


class _Prices:
    def __init__(self, prices: pd.DataFrame):
        self.series = {s: g.sort_values("date").set_index("date")["close"]
                       for s, g in prices.groupby("symbol")}

    def close(self, symbol, day) -> float:
        s = self.series.get(symbol)
        if s is None:
            return 0.0
        s = s[s.index <= day]
        return float(s.iloc[-1]) if len(s) else 0.0


def _apply(u: _Unit, etype: str, qty: int, price: float) -> None:
    d = m.DIRECTION[etype]
    if d > 0:
        cost = float(price) if etype in m.COST_INFLOWS else 0.0
        new_avg = cost if u.balance <= 0 else (u.balance * u.avg + qty * cost) / (u.balance + qty)
        u.balance += qty
        u.avg = round(new_avg, 4) if u.balance > 0 else 0.0
    elif d < 0:
        if etype in m.REALISING_OUTFLOWS:
            u.realized += (float(price) - u.avg) * qty
        elif etype == m.RIGHTS_EXPIRE:
            u.realized -= u.avg * qty
        u.balance -= qty
        if u.balance <= 0:
            u.avg = 0.0
    key = {m.BUY: "bought_total", m.OPTION_SETTLE_IN: "bought_total", m.SELL: "sold_total",
           m.OPTION_SETTLE_OUT: "sold_total", m.BONUS_ACTIVATE: "bonus_total",
           m.RIGHTS_INIT: "rights_total", m.RIGHTS_CONVERT: "rights_total",
           m.CUSTODIAN_IN: "transfer_in_total", m.CUSTODIAN_OUT: "transfer_out_total"}.get(etype)
    if key:
        u.totals[key] += qty


def build_chain2(source_events: pd.DataFrame, corporate_actions: pd.DataFrame,
                 symbols: pd.DataFrame, prices: pd.DataFrame, calendar: m.Calendar,
                 depository_daily: pd.DataFrame | None = None, lag_days: int = 2,
                 customers=None) -> Chain2:
    """Replay source events and corporate actions into the ledger and daily positions.

    Entitlement is the balance at the close of the business day before the record date
    (corporate-action rows are stamped 08:15, before trading). When ``depository_daily``
    is given, a gap between the end-of-day balance and the depository quantity that is not
    explained by a settlement lag of up to ``lag_days`` is closed with an UNEXPLAINED plug.
    """
    src = source_events.copy()
    if customers is not None:
        src = src[src["customer_id"].isin(set(customers))]
    src = src.sort_values(["customer_id", "date", "time", "source_ref"])
    klass = dict(zip(symbols["symbol"], symbols["instrument_class"]))
    px = _Prices(prices)
    cas = corporate_actions.copy()
    by_record, by_activation, by_expiry, rights_ca = defaultdict(list), defaultdict(list), \
        defaultdict(list), {}
    for ca in cas.itertuples(index=False):
        by_record[(ca.symbol, ca.record_date)].append(ca)
        if pd.notna(ca.activation_date):
            by_activation[(ca.symbol, ca.activation_date)].append(ca)
        if ca.action_type == m.RIGHTS:
            rights_ca[ca.rights_symbol] = ca
            by_expiry[(ca.rights_symbol, ca.period_end)].append(ca)

    dep = defaultdict(dict)
    dep_customers = set()
    if depository_daily is not None and len(depository_daily):
        live = depository_daily[depository_daily["row_type"] != m.ROW_IGNORE]
        for r in live.itertuples(index=False):
            if customers is None or r.customer_id in set(customers):
                dep[(r.customer_id, r.symbol)][r.date] = int(r.quantity)
                dep_customers.add(r.customer_id)

    events_out, daily_out = [], []
    all_customers = sorted(set(src["customer_id"]) | dep_customers)
    for cust in all_customers:
        cs = src[src["customer_id"] == cust]
        src_by = defaultdict(list)
        for r in cs.itertuples(index=False):
            src_by[(r.symbol, r.date)].append(r)
        units: dict[str, _Unit] = {}
        first = [d for d in cs["date"]] + [d for (c, s), v in dep.items() if c == cust for d in v]
        start = min(first)
        for sym in set(cs["symbol"]) | {s for (c, s) in dep if c == cust}:
            units[sym] = _Unit()
        exercised = defaultdict(lambda: [0, 0.0])   # ca_id -> [quantity, cost of rights]
        pending_init = defaultdict(list)             # (rights_symbol, day) -> [(ca, qty)]

        def emit(sym, day, etype, qty, price, time, origin=m.ORIGIN_DERIVED, source_ref=None,
                 ca=None, ca_value=None, cash=None, deleted=False):
            u = units[sym]
            if deleted:      # stored as it was calculated before the soft delete, not applied
                phantom = _Unit(balance=u.balance, avg=u.avg)
                _apply(phantom, etype, qty, price)
                bal, avg = phantom.balance, phantom.avg
            else:
                _apply(u, etype, qty, price)
                bal, avg = u.balance, u.avg
            u.events.append(dict(customer_id=cust, symbol=sym, date=day, time=time,
                                 event_type=etype, origin=origin, source_ref=source_ref,
                                 quantity=int(qty), price=float(price), balance_after=bal,
                                 avg_cost_after=avg, ca_id=getattr(ca, "ca_id", None),
                                 ca_value=ca_value, cash_amount=cash, is_deleted=int(deleted),
                                 calc_status=m.CALCULATED))

        for day in calendar:
            if day < start:
                continue
            mains = sorted(s for s in units if klass.get(s) != m.RIGHT)
            for sym in mains + ["__rights__"]:
                todo = sorted(s for s in units if klass.get(s) == m.RIGHT) \
                    if sym == "__rights__" else [sym]
                for s in todo:
                    _process_day(cust, s, day, units, emit, by_record, by_activation, by_expiry,
                                 rights_ca, src_by, exercised, pending_init, px, dep, calendar,
                                 lag_days)
        for sym, u in units.items():
            events_out.extend(u.events)
            daily_out.extend(u.daily)

    events = pd.DataFrame(events_out, columns=EVENT_COLUMNS[1:])
    events = events.sort_values(["customer_id", "symbol", "date", "time"], kind="stable")
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    events = events.reset_index(drop=True)
    daily = pd.DataFrame(daily_out, columns=DAILY_COLUMNS).sort_values(
        ["customer_id", "symbol", "date"]).reset_index(drop=True)
    latest = daily.groupby(["customer_id", "symbol"], sort=True).tail(1).reset_index(drop=True)
    return Chain2(events=events, daily=daily, latest=latest)


def _process_day(cust, sym, day, units, emit, by_record, by_activation, by_expiry, rights_ca,
                 src_by, exercised, pending_init, px, dep, calendar, lag_days):
    u = units[sym]
    n_before = len(u.events)
    start_balance = u.balance
    # 1. record-date corporate actions on this symbol
    for ca in by_record.get((sym, day), []):
        if start_balance <= 0:
            continue
        if ca.action_type == m.DIVIDEND:
            emit(sym, day, m.CASH_DIVIDEND, 0, 0.0, CA_TIME, ca=ca,
                 ca_value=float(ca.value), cash=round(start_balance * float(ca.value), 2))
        elif ca.action_type == m.BONUS:
            q = m.entitlement(start_balance, ca.value)
            if q > 0:
                emit(sym, day, m.BONUS_GRANT, q, 0.0, CA_TIME, ca=ca, ca_value=float(ca.value))
                u.pending_bonus[ca.ca_id] = u.pending_bonus.get(ca.ca_id, 0) + q
        elif ca.action_type == m.RIGHTS:
            q = m.entitlement(start_balance, ca.value)
            if q > 0:
                emit(sym, day, m.RIGHTS_GRANT, q, 0.0, CA_TIME, ca=ca, ca_value=float(ca.value))
                pending_init[(ca.rights_symbol, day)].append((ca, q))
                units.setdefault(ca.rights_symbol, _Unit())
    # rights credited on the rights symbol on the record date
    for ca, q in pending_init.pop((sym, day), []):
        emit(sym, day, m.RIGHTS_INIT, q, 0.0, CA_TIME, ca=ca, ca_value=float(ca.value))
    # 2. activation date
    for ca in by_activation.get((sym, day), []):
        if ca.action_type == m.BONUS:
            q = u.pending_bonus.pop(ca.ca_id, 0)
            if q > 0:
                emit(sym, day, m.BONUS_ACTIVATE, q, 0.0, ACTIVATION_TIME, ca=ca,
                     ca_value=float(ca.value))
        elif ca.action_type == m.RIGHTS:
            q, cost = exercised.get(ca.ca_id, (0, 0.0))
            if q > 0:
                price = round(m.SUBSCRIPTION_PRICE + cost / q, 4)
                emit(sym, day, m.RIGHTS_CONVERT, q, price, ACTIVATION_TIME, ca=ca,
                     ca_value=float(ca.value))
    # 3. source events in time order
    for ev in src_by.get((sym, day), []):
        deleted = bool(ev.is_deleted)
        price = ev.price
        if ev.event_type == m.CUSTODIAN_IN and (price is None or pd.isna(price) or price == 0):
            price = px.close(sym, day)
        if ev.event_type == m.RIGHTS_EXERCISE and not deleted:
            ca = rights_ca.get(sym)
            if ca is not None:
                exercised[ca.ca_id][0] += int(ev.quantity)
                exercised[ca.ca_id][1] += u.avg * int(ev.quantity)
        emit(sym, day, ev.event_type, int(ev.quantity), float(price or 0.0), ev.time,
             origin=m.ORIGIN_SOURCE, source_ref=ev.source_ref, deleted=deleted)
    # 4. rights lapse at the end of the subscription period
    for ca in by_expiry.get((sym, day), []):
        if u.balance > 0:
            emit(sym, day, m.RIGHTS_EXPIRE, u.balance, 0.0, EXPIRY_TIME, ca=ca,
                 ca_value=float(ca.value))
    # 5. gap plug against the depository, unless a settlement lag explains the difference
    q_dep = dep.get((cust, sym), {}).get(day) if dep else None
    if q_dep is not None and u.balance != q_dep:
        lagged = any(u.eod.get(calendar.shift(day, -k)) == q_dep for k in range(1, lag_days + 1))
        if not lagged:
            diff = q_dep - u.balance
            etype = m.UNEXPLAINED_INCREASE if diff > 0 else m.UNEXPLAINED_DECREASE
            emit(sym, day, etype, abs(diff), px.close(sym, day), PLUG_TIME)
    u.eod[day] = u.balance
    if any(not e["is_deleted"] for e in u.events[n_before:]):
        u.daily.append(dict(customer_id=cust, symbol=sym, date=day, quantity=u.balance,
                            **u.totals, avg_cost=u.avg, realized_pnl=round(u.realized, 2),
                            calc_version=1))


# ======================================================================================
# Downstream: return engine and app display
# ======================================================================================
def holdings_asof(daily: pd.DataFrame, day) -> pd.DataFrame:
    """Quantity per customer and symbol from the last daily row on or before ``day``."""
    d = daily[daily["date"] <= day].sort_values("date")
    return d.groupby(["customer_id", "symbol"]).tail(1)[["customer_id", "symbol", "quantity"]]


def last_close(prices: pd.DataFrame, day) -> pd.Series:
    p = prices[prices["date"] <= day].sort_values("date")
    return p.groupby("symbol")["close"].last()


def portfolio_values(daily: pd.DataFrame, prices: pd.DataFrame, day) -> pd.DataFrame:
    """Gross market value per customer at ``day``; flags customers holding an unpriced symbol."""
    h = holdings_asof(daily, day)
    h = h[h["quantity"] != 0].copy()
    h["close"] = h["symbol"].map(last_close(prices, day))
    h["value"] = h["quantity"] * h["close"]
    g = h.groupby("customer_id").agg(value=("value", "sum"),
                                     unpriced=("close", lambda s: int(s.isna().sum())))
    return g


def build_return_outputs(customers: pd.DataFrame, daily: pd.DataFrame, prices: pd.DataFrame,
                         calendar: m.Calendar, horizons=None) -> pd.DataFrame:
    horizons = horizons or HORIZONS
    as_of = calendar.as_of
    end = portfolio_values(daily, prices, as_of)
    rows = []
    for label, n in horizons:
        start_day = calendar.shift(as_of, -n)
        start = portfolio_values(daily, prices, start_day)
        for cust in customers["customer_id"]:
            sv = start["value"].get(cust, 0.0)
            ev = end["value"].get(cust, 0.0)
            unpriced = start["unpriced"].get(cust, 0) + end["unpriced"].get(cust, 0)
            if unpriced:
                status, sv, ev = "MISSING_PRICE", None, None
            elif sv == 0 and ev == 0:
                status = "NO_HOLDINGS"
            elif sv < 0 or ev < 0:
                status = "INVALID_START_VALUE"
            else:
                status = "OK"
            rows.append(dict(customer_id=cust, as_of=as_of, horizon=label, status=status,
                             start_value=None if sv is None else round(sv),
                             end_value=None if ev is None else round(ev)))
    return pd.DataFrame(rows)


def build_app_display(latest: pd.DataFrame, symbols: pd.DataFrame, fees: pd.DataFrame,
                      prices: pd.DataFrame, as_of, fee_override: dict | None = None) -> pd.DataFrame:
    """What the app shows, using the display formula chain:

    market value = shares x close x (1 - sell fee); cost = shares x average cost;
    unrealized P&L = market value - cost; return = P&L / cost;
    break-even price = average cost x (1 + sell fee).
    """
    klass = dict(zip(symbols["symbol"], symbols["instrument_class"]))
    fee = dict(zip(fees["instrument_class"], fees["sell_fee"]))
    fee.update(fee_override or {})
    close = last_close(prices, as_of)
    rows = []
    for r in latest[latest["quantity"] > 0].itertuples(index=False):
        f = fee.get(klass.get(r.symbol), 0.0)
        c = close.get(r.symbol)
        cost = round(r.quantity * r.avg_cost)
        if c is None or pd.isna(c):
            mv = pnl = ret = None
        else:
            mv = round(r.quantity * c * (1 - f))
            pnl = mv - cost
            ret = round(pnl / cost, 6) if cost else None
        rows.append(dict(customer_id=r.customer_id, symbol=r.symbol, as_of=as_of,
                         quantity=int(r.quantity), close=None if c is None or pd.isna(c) else c,
                         market_value=mv, cost_value=cost, unrealized_pnl=pnl, return_pct=ret,
                         break_even_price=round(r.avg_cost * (1 + f), 2)))
    return pd.DataFrame(rows)
