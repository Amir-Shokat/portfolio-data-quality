"""Synthetic data: a clean world, then labelled defects injected on purpose.

``make_dataset(seed, defects)`` builds 60 trading days for 40 fictional customers and seven
fictional symbols, runs the reference pipeline, and injects the requested defects at the
stage where they happen in real life (price feed, depository feed, trade feed, stored
tables, downstream consumers). Every defect declares which checks must fire, so the tests
can show that each check catches exactly what it should and nothing on clean data.

All names, identifiers and amounts are invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import model as m
from . import pipeline as p
from .dataset import Dataset

START = "2026-04-04"
N_DAYS = 60

SYMBOLS = [
    # symbol, ISIN body (11 chars), name, class, parent, base price, daily vol, drift
    ("ALPHA", "IRO1ALPH000", "Alpha Industries", m.EQUITY, None, 12_000, 0.018, 0.0004),
    ("BETA", "IRO1BETA000", "Beta Petrochemical", m.EQUITY, None, 8_000, 0.020, 0.0003),
    ("GAMMA", "IRO1GAMM000", "Gamma Cement", m.EQUITY, None, 15_000, 0.017, 0.0005),
    ("GAMMA_R", "IRR1GAMM010", "Gamma Cement rights", m.RIGHT, "GAMMA", None, None, None),
    ("DELTA", "IRO1DELT000", "Delta Steel", m.EQUITY, None, 5_000, 0.022, 0.0002),
    ("EPSIL", "IRO1EPSL000", "Epsilon Foods", m.EQUITY, None, 22_000, 0.015, 0.0003),
    ("FIXED", "IRT1FXDA000", "Fixed Income Fund A", m.FIXED_INCOME, None, 18_000, 0.0005, 0.0008),
]
DELTA_OLD_BODY = "IRO3DELT000"     # DELTA moved market on day 20
FEES = [(m.EQUITY, 0.0088, 0.05), (m.RIGHT, 0.0088, 0.10), (m.FIXED_INCOME, 0.0004, 0.03)]

# (ca_id, symbol, type, session day, record day, activation day, period end day, value, rights)
CORPORATE_ACTIONS = [
    ("CA01", "ALPHA", m.DIVIDEND, 15, 18, None, None, 600.0, None),
    ("CA02", "BETA", m.BONUS, 20, 24, 35, None, 0.5, None),
    ("CA03", "GAMMA", m.RIGHTS, 28, 28, 50, 40, 0.4, "GAMMA_R"),
]

# (customer, symbol, day, event type, quantity, time, price override, deleted)
SCHEDULE = [
    # C001-C011: scenario customers (one reconciliation scenario each)
    ("C001", "EPSIL", 2, m.BUY, 300, "09:10:00"), ("C001", "EPSIL", 10, m.BUY, 200, "10:00:00"),
    ("C001", "EPSIL", 30, m.SELL, 300, "11:00:00"), ("C001", "FIXED", 3, m.BUY, 500, "09:30:00"),
    ("C001", "FIXED", 44, m.SELL, 200, "10:30:00"),
    ("C002", "ALPHA", 3, m.BUY, 1000, "09:20:00"), ("C002", "ALPHA", 25, m.SELL, 400, "10:40:00"),
    ("C003", "BETA", 5, m.BUY, 1234, "09:40:00"), ("C003", "BETA", 30, m.SELL, 234, "11:15:00"),
    ("C004", "GAMMA", 8, m.BUY, 2000, "09:05:00"), ("C004", "GAMMA_R", 32, m.SELL, 200, "10:00:00"),
    ("C004", "GAMMA_R", 36, m.RIGHTS_EXERCISE, 500, "09:00:00"),
    ("C005", "DELTA", 4, m.BUY, 600, "10:20:00"),
    ("C006", "ALPHA", 6, m.BUY, 500, "09:50:00"), ("C006", "ALPHA", 26, m.CUSTODIAN_OUT, 500, "09:00:00"),
    ("C006", "ALPHA", 33, m.CUSTODIAN_IN, 300, "09:00:00"),
    ("C007", "BETA", 7, m.BUY, 800, "10:10:00"),
    ("C008", "FIXED", 9, m.BUY, 300, "09:35:00"), ("C008", "FIXED", 27, m.BUY, 150, "10:00:00"),
    ("C009", "GAMMA", 12, m.OPTION_SETTLE_IN, 1000, "09:30:00", 14_000),
    ("C009", "GAMMA", 12, m.SELL, 300, "11:30:00"),
    ("C010", "EPSIL", 14, m.BUY, 500, "10:00:00", None, True),
    ("C010", "EPSIL", 14, m.BUY, 450, "10:05:00"), ("C010", "EPSIL", 40, m.SELL, 100, "11:00:00"),
    ("C011", "DELTA", 22, m.BUY, 800, "09:15:00"), ("C011", "DELTA", 52, m.SELL, 300, "10:45:00"),
    # C012-C030: one defect target each
    ("C012", "ALPHA", 10, m.BUY, 700, "09:00:00"), ("C012", "ALPHA", 30, m.SELL, 700, "10:00:00"),
    ("C012", "ALPHA", 44, m.BUY, 300, "09:45:00"),
    ("C013", "BETA", 3, m.BUY, 400, "10:00:00"),
    ("C014", "FIXED", 5, m.BUY, 900, "09:00:00"), ("C014", "FIXED", 21, m.BUY, 300, "10:00:00"),
    ("C014", "FIXED", 47, m.SELL, 500, "11:00:00"),
    ("C015", "DELTA", 6, m.BUY, 1000, "09:00:00"), ("C015", "DELTA", 49, m.SELL, 400, "10:00:00"),
    ("C015", "DELTA", 55, m.BUY, 100, "11:00:00"),
    ("C016", "ALPHA", 12, m.BUY, 250, "09:30:00"), ("C016", "ALPHA", 33, m.BUY, 250, "10:30:00"),
    ("C017", "BETA", 2, m.BUY, 600, "09:00:00"), ("C017", "BETA", 16, m.BUY, 200, "10:00:00"),
    ("C017", "BETA", 45, m.SELL, 300, "11:00:00"),
    ("C018", "ALPHA", 4, m.BUY, 900, "09:00:00"), ("C018", "ALPHA", 38, m.SELL, 100, "10:00:00"),
    ("C019", "GAMMA", 9, m.BUY, 1500, "09:00:00"),
    ("C020", "EPSIL", 6, m.BUY, 400, "09:00:00"), ("C020", "EPSIL", 51, m.SELL, 150, "10:00:00"),
    ("C021", "FIXED", 3, m.BUY, 1200, "09:00:00"), ("C021", "FIXED", 20, m.SELL, 300, "10:00:00"),
    ("C021", "FIXED", 41, m.BUY, 100, "11:00:00"),
    ("C022", "DELTA", 8, m.BUY, 700, "09:00:00"), ("C022", "DELTA", 8, m.BUY, 300, "11:00:00"),
    ("C022", "DELTA", 33, m.SELL, 200, "10:00:00"),
    ("C023", "BETA", 26, m.BUY, 500, "09:00:00"), ("C023", "BETA", 39, m.BUY, 300, "10:00:00"),
    ("C023", "BETA", 53, m.SELL, 200, "11:00:00"),
    ("C024", "ALPHA", 9, m.BUY, 600, "09:00:00"), ("C024", "ALPHA", 31, m.SELL, 200, "10:00:00"),
    ("C025", "GAMMA", 30, m.BUY, 800, "09:00:00"), ("C025", "GAMMA", 43, m.SELL, 100, "10:00:00"),
    ("C025", "GAMMA", 56, m.BUY, 200, "11:00:00"),
    ("C026", "EPSIL", 11, m.BUY, 700, "09:00:00"), ("C026", "EPSIL", 54, m.SELL, 250, "10:00:00"),
    ("C027", "BETA", 13, m.BUY, 350, "09:00:00"), ("C027", "BETA", 41, m.SELL, 100, "10:00:00"),
    ("C028", "DELTA", 15, m.BUY, 900, "09:00:00"), ("C028", "FIXED", 19, m.BUY, 120, "10:00:00"),
    ("C029", "FIXED", 18, m.BUY, 400, "09:00:00"),
    ("C030", "ALPHA", 7, m.BUY, 300, "09:00:00"),
]
RANDOM_CUSTOMERS = [f"C{i:03d}" for i in range(31, 41)]
TRADABLE = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSIL", "FIXED"]


def variant_isin(isin: str) -> str:
    """Same first 11 characters, different (invalid) 12th character: a known variant issue."""
    return isin[:11] + str((int(isin[11]) + 3) % 10)


@dataclass
class World:
    """Everything the generator builds, stage by stage. Defects mutate it in place."""
    calendar: m.Calendar
    customers: pd.DataFrame
    symbols: pd.DataFrame
    lineage: pd.DataFrame
    fees: pd.DataFrame
    prices: pd.DataFrame
    corporate_actions: pd.DataFrame
    source_events: pd.DataFrame
    feed: pd.DataFrame = None
    chain1_daily: pd.DataFrame = None
    chain1_latest: pd.DataFrame = None
    events: pd.DataFrame = None
    daily: pd.DataFrame = None
    latest: pd.DataFrame = None
    cash_movements: pd.DataFrame = None
    return_outputs: pd.DataFrame = None
    return_cash_flows: pd.DataFrame = None
    app_display: pd.DataFrame = None
    rng: np.random.Generator = None

    def day(self, i: int) -> pd.Timestamp:
        return self.calendar.days[i]


# ======================================================================================
# Defect registry
# ======================================================================================
STAGES = ("source", "feed", "trade_feed", "stored", "downstream")


@dataclass(frozen=True)
class DefectSpec:
    name: str
    stage: str
    expected_checks: frozenset
    description: str
    benign: bool
    apply: Callable[[World], str] = field(compare=False)


DEFECTS: dict[str, DefectSpec] = {}


def defect(name, stage, expected, description, benign=False):
    def wrap(fn):
        DEFECTS[name] = DefectSpec(name, stage, frozenset(expected), description, benign, fn)
        return fn
    return wrap


def _rows(df, **eq):
    mask = pd.Series(True, index=df.index)
    for k, v in eq.items():
        mask &= df[k] == v
    return mask


# ---- price feed ----------------------------------------------------------------------
@defect("price_missing", "source", {"B1"}, "The price feed never covered EPSIL, a held symbol.")
def _price_missing(w):
    w.prices = w.prices[w.prices["symbol"] != "EPSIL"].reset_index(drop=True)
    return "EPSIL prices"


@defect("price_stale", "source", {"B2"}, "DELTA prices stopped five trading days before the as-of date.")
def _price_stale(w):
    w.prices = w.prices[~((w.prices["symbol"] == "DELTA") & (w.prices["date"] > w.day(54)))]
    return "DELTA prices after day 54"


@defect("price_spike", "source", {"B3"}, "One ALPHA close is 30% above the previous close, then falls back.")
def _price_spike(w):
    i = _rows(w.prices, symbol="ALPHA", date=w.day(45))
    w.prices.loc[i, "close"] = (w.prices.loc[i, "close"] * 1.30).round()
    return "ALPHA close on day 45"


# ---- depository feed -----------------------------------------------------------------
@defect("settlement_lag", "feed", set(), "C014's depository rows arrive one trading day after the trades "
        "(settlement lag). Benign: must not be reported as an exception.", benign=True)
def _settlement_lag(w):
    i = w.feed["customer_id"] == "C014"
    w.feed.loc[i, "date"] = w.feed.loc[i, "date"].map(lambda d: w.calendar.shift(d, 1))
    return "C014 FIXED feed rows"


@defect("depository_feed_stopped", "feed", {"B6", "A2X_LATEST", "A2X_HIST"},
        "The depository feed stopped updating C015/DELTA after day 45 while trades continued.")
def _feed_stopped(w):
    w.feed = w.feed[~((w.feed["customer_id"] == "C015") & (w.feed["date"] > w.day(45)))]
    return "C015 DELTA feed rows after day 45"


@defect("unexplained_admin_increase", "feed", {"A1"},
        "A manual (ADMIN) depository registration adds 500 DELTA shares for C011 with no trade "
        "behind it; the ledger closes the gap with an UNEXPLAINED_INCREASE plug.")
def _admin_increase(w):
    f = w.feed
    i = (f["customer_id"] == "C011") & (f["date"] >= w.day(42))
    f.loc[i, "quantity"] += 500
    held = f[(f["customer_id"] == "C011") & (f["date"] < w.day(42))].sort_values("date")
    qty = int(held["quantity"].iloc[-1]) + 500
    row = dict(feed_id="F_ADMIN", customer_id="C011", isin=f.loc[held.index[-1], "isin"],
               date=w.day(42), quantity=qty, origin=m.FEED_ADMIN)
    w.feed = pd.concat([f, pd.DataFrame([row])], ignore_index=True)
    return "C011 DELTA, day 42"


# ---- trade feed ----------------------------------------------------------------------
@defect("oversell_plugged", "trade_feed", {"A0", "A1"},
        "The trade feed delivered C012's full-exit sell twice. The ledger goes negative and the "
        "gap mechanism plugs it back to the depository quantity the same day.")
def _oversell(w):
    s = w.source_events
    i = s.index[_rows(s, customer_id="C012", symbol="ALPHA", event_type=m.SELL)][0]
    dup = s.loc[[i]].copy()
    dup["time"] = "10:01:00"
    dup["source_ref"] = dup["source_ref"] + "X"
    w.source_events = pd.concat([s, dup], ignore_index=True)
    return "C012 ALPHA, day 30"


@defect("intraday_negative", "trade_feed", {"A0"},
        "A sell carries an earlier timestamp than the buy that funded it, so C013's BETA balance "
        "is negative for an hour; the day closes correctly (masked at the daily rollup).")
def _intraday_negative(w):
    base = dict(customer_id="C013", symbol="BETA", date=w.day(46), quantity=900, is_deleted=0)
    close = float(w.prices[_rows(w.prices, symbol="BETA", date=w.day(46))]["close"].iloc[0])
    rows = [dict(base, event_type=m.SELL, time="10:00:00", price=close, source_ref="T_NEG1"),
            dict(base, event_type=m.BUY, time="11:00:00", price=close, source_ref="T_NEG2")]
    w.source_events = pd.concat([w.source_events, pd.DataFrame(rows)], ignore_index=True)
    return "C013 BETA, day 46"


# ---- stored tables -------------------------------------------------------------------
@defect("fragmented_security", "stored", {"A11"},
        "ALPHA also exists as ALPHA_V (same first 11 ISIN characters); C016's depository rows "
        "are stored under ALPHA_V while its ledger uses ALPHA.")
def _fragmented(w):
    alpha = w.symbols[w.symbols["symbol"] == "ALPHA"].iloc[0]
    extra = dict(symbol="ALPHA_V", isin=variant_isin(alpha["isin"]), name=alpha["name"],
                 instrument_class=m.EQUITY, parent_symbol=None)
    w.symbols = pd.concat([w.symbols, pd.DataFrame([extra])], ignore_index=True)
    for t in ("chain1_daily", "chain1_latest"):
        df = getattr(w, t)
        df.loc[_rows(df, customer_id="C016", symbol="ALPHA"), "symbol"] = "ALPHA_V"
    return "C016 ALPHA depository rows"


@defect("duplicate_daily_row", "stored", {"A9"}, "One positions_daily row for C017/BETA is stored twice.")
def _duplicate_daily(w):
    d = w.daily
    i = d.index[_rows(d, customer_id="C017", symbol="BETA", date=w.day(16))]
    w.daily = pd.concat([d, d.loc[i]], ignore_index=True)
    return "C017 BETA, day 16"


@defect("orphan_ca_reference", "stored", {"A10"},
        "An extra dividend row for C018 points to a corporate action id that does not exist.")
def _orphan_ca(w):
    e = w.events
    prev = e[_rows(e, customer_id="C018", symbol="ALPHA") & (e["date"] < w.day(38))].iloc[-1]
    row = prev.to_dict()
    row.update(event_id=int(e["event_id"].max()) + 1, date=w.day(38), time=p.CA_TIME,
               event_type=m.CASH_DIVIDEND, origin=m.ORIGIN_DERIVED, source_ref=None, quantity=0,
               price=0.0, ca_id="CA99", ca_value=450.0,
               cash_amount=float(prev["balance_after"]) * 450.0)
    w.events = pd.concat([e, pd.DataFrame([row])], ignore_index=True)
    return "C018 ALPHA, day 38"


@defect("depository_snapshot_mismatch", "stored", {"A2W", "A2X_LATEST"},
        "C019's depository snapshot shows 10 more GAMMA shares than its own latest history row.")
def _dep_snapshot(w):
    w.chain1_latest.loc[_rows(w.chain1_latest, customer_id="C019", symbol="GAMMA"), "quantity"] += 10
    return "C019 GAMMA depository_latest"


@defect("unlinked_isin_change", "stored", {"A8"},
        "The ISIN lineage table lost the link for DELTA's market move, so received old ISINs "
        "can no longer be explained.")
def _unlinked(w):
    w.lineage = w.lineage.iloc[0:0].copy()
    return "isin_lineage"


@defect("snapshot_stale_column", "stored", {"A3"},
        "C020's snapshot updated the quantity after a sale but kept the old realized P&L.")
def _stale_column(w):
    d = w.daily[_rows(w.daily, customer_id="C020", symbol="EPSIL")]
    w.latest.loc[_rows(w.latest, customer_id="C020", symbol="EPSIL"), "realized_pnl"] = \
        d["realized_pnl"].iloc[-2]
    return "C020 EPSIL positions_latest.realized_pnl"


@defect("daily_rollup_break", "stored", {"A4", "A2X_HIST"},
        "The daily rollup stored 50 fewer FIXED shares for C021 on day 20 than the ledger shows.")
def _rollup_break(w):
    w.daily.loc[_rows(w.daily, customer_id="C021", symbol="FIXED", date=w.day(20)), "quantity"] -= 50
    return "C021 FIXED, day 20"


@defect("ledger_balance_break", "stored", {"A5B"},
        "One ledger row for C022 carries a running balance 100 shares too high.")
def _balance_break(w):
    i = _rows(w.events, customer_id="C022", symbol="DELTA", date=w.day(8), time="09:00:00")
    w.events.loc[i, "balance_after"] += 100
    return "C022 DELTA, day 8 09:00"


@defect("avg_cost_jump", "stored", {"A6"}, "A sell changed C023's average cost, which a sale cannot do.")
def _avg_jump(w):
    i = w.events.index[_rows(w.events, customer_id="C023", symbol="BETA", event_type=m.SELL)][-1]
    w.events.loc[i, "avg_cost_after"] = round(w.events.loc[i, "avg_cost_after"] * 1.1, 4)
    return "C023 BETA, day 53"


@defect("missing_dividend_event", "stored", {"A12"},
        "C024 held ALPHA on the record date but never received the cash-dividend row.")
def _missing_dividend(w):
    w.events = w.events[~_rows(w.events, customer_id="C024", event_type=m.CASH_DIVIDEND)]
    w.daily = w.daily[~_rows(w.daily, customer_id="C024", symbol="ALPHA", date=w.day(18))]
    return "C024 ALPHA, CA01"


@defect("version_drift", "stored", {"B8"},
        "C025's GAMMA series switches calculation version for its last two days.")
def _version_drift(w):
    i = w.daily.index[_rows(w.daily, customer_id="C025", symbol="GAMMA")][-2:]
    w.daily.loc[i, "calc_version"] = 2
    w.latest.loc[_rows(w.latest, customer_id="C025", symbol="GAMMA"), "calc_version"] = 2
    return "C025 GAMMA"


@defect("calc_stuck", "stored", {"B4", "B6", "A2X_LATEST", "A2X_HIST"},
        "C026's sale on day 54 sits in the calculation queue (PENDING): no balance, no daily row, "
        "stale snapshot. The depository already shows the sale.")
def _calc_stuck(w):
    i = _rows(w.events, customer_id="C026", symbol="EPSIL", date=w.day(54))
    w.events.loc[i, ["calc_status"]] = m.PENDING
    w.events.loc[i, ["balance_after", "avg_cost_after"]] = np.nan
    w.daily = w.daily[~_rows(w.daily, customer_id="C026", symbol="EPSIL", date=w.day(54))]
    prev = w.daily[_rows(w.daily, customer_id="C026", symbol="EPSIL")].iloc[-1]
    j = w.latest.index[_rows(w.latest, customer_id="C026", symbol="EPSIL")]
    for c in w.latest.columns:
        w.latest.loc[j, c] = prev[c]
    return "C026 EPSIL, day 54"


@defect("registration_type_unset", "stored", set(),
        "C027's depository rows lost their registration type (informational field only). Benign "
        "for the catalogue; the rebuild reports it as Info.", benign=True)
def _reg_unset(w):
    w.chain1_daily.loc[_rows(w.chain1_daily, customer_id="C027"), "registration_type"] = None
    return "C027 depository_daily.registration_type"


@defect("leave_rejoin_not_ignored", "stored", set(),
        "C005's spurious leave-and-return row was not marked IGNORE. Within tolerance it looks "
        "like a one-day timing gap, so no check fires; only the rebuild catches it (scenario S07).")
def _leave_not_ignored(w):
    w.chain1_daily.loc[_rows(w.chain1_daily, customer_id="C005", row_type=m.ROW_IGNORE),
                       "row_type"] = m.ROW_LEAVE
    return "C005 DELTA, day 22"


# ---- downstream ----------------------------------------------------------------------
@defect("display_fee_constant", "downstream", {"C1"},
        "The app applies the equity sell fee to the fixed-income fund.")
def _fee_constant(w):
    equity_fee = float(w.fees.loc[w.fees["instrument_class"] == m.EQUITY, "sell_fee"].iloc[0])
    w.app_display = p.build_app_display(w.latest, w.symbols, w.fees, w.prices, w.calendar.as_of,
                                        fee_override={m.FIXED_INCOME: equity_fee})
    return "app_display, FIXED rows"


@defect("return_value_mismatch", "downstream", {"C3"},
        "The return engine's end value for C028 is 5% above the value of its holdings.")
def _return_value(w):
    i = _rows(w.return_outputs, customer_id="C028")
    w.return_outputs.loc[i, "end_value"] = (w.return_outputs.loc[i, "end_value"] * 1.05).round()
    return "C028 return_outputs"


@defect("unknown_status_code", "downstream", {"C4"},
        "One return row carries a status code that is not on the known list.")
def _unknown_status(w):
    w.return_outputs.loc[_rows(w.return_outputs, customer_id="C029", horizon="1M"), "status"] = "E_TIMEOUT"
    return "C029 1M"


@defect("dividend_counted_as_flow", "downstream", {"C5"},
        "The return engine counted C030's dividend credit as an external deposit.")
def _dividend_flow(w):
    c = w.cash_movements[_rows(w.cash_movements, customer_id="C030", kind="DIVIDEND_CREDIT")]
    w.return_cash_flows = pd.concat([w.return_cash_flows, c[["customer_id", "date", "amount"]]],
                                    ignore_index=True)
    return "C030 dividend credit"


# ======================================================================================
# Building the clean world
# ======================================================================================
def _prices(calendar: m.Calendar, rng: np.random.Generator, fees) -> pd.DataFrame:
    limits = {k: lim for k, _, lim in fees}
    ca_by_day = {}
    for ca_id, sym, typ, _, rec, *_rest, value, _r in CORPORATE_ACTIONS:
        ca_by_day.setdefault((sym, rec), []).append((typ, value))
    rows, series = [], {}
    for sym, _, _, klass, parent, base, vol, drift in SYMBOLS:
        if parent:
            continue
        lim = 0.8 * limits[klass]
        close = float(base)
        for i, day in enumerate(calendar):
            if i:
                b = m.adjusted_base_price(close, ca_by_day.get((sym, i), []))
                r = float(np.clip(rng.normal(drift, vol), -lim, lim))
                close = float(round(b * (1 + r)))
            series[(sym, i)] = close
            rows.append(dict(symbol=sym, date=day, close=close))
    rights = next(c for c in CORPORATE_ACTIONS if c[2] == m.RIGHTS)
    for i in range(rights[4], rights[6] + 1):     # rights trade from record date to period end
        theo = max(series[("GAMMA", i)] - m.SUBSCRIPTION_PRICE, 100)
        rows.append(dict(symbol="GAMMA_R", date=calendar.days[i],
                         close=float(round(theo * (1 + rng.uniform(-0.01, 0.01))))))
    return pd.DataFrame(rows)


def _random_schedule(rng: np.random.Generator) -> list[tuple]:
    out = []
    for cust in RANDOM_CUSTOMERS:
        syms = rng.choice(TRADABLE, size=int(rng.integers(2, 4)), replace=False)
        for sym in syms:
            d0 = int(rng.integers(0, 30))
            bal = int(rng.integers(2, 60)) * 50
            out.append((cust, str(sym), d0, m.BUY, bal, f"{int(rng.integers(9, 11)):02d}:"
                        f"{int(rng.integers(0, 60)):02d}:00"))
            for d in range(d0 + 1, 58):
                if rng.random() >= 0.07:
                    continue
                t = f"{int(rng.integers(9, 11)):02d}:{int(rng.integers(0, 60)):02d}:00"
                if bal > 0 and rng.random() < 0.45:
                    q = bal if rng.random() < 0.15 else min(bal, int(rng.integers(1, 20)) * 50)
                    out.append((cust, str(sym), d, m.SELL, q, t))
                    bal -= q
                else:
                    q = int(rng.integers(1, 30)) * 50
                    out.append((cust, str(sym), d, m.BUY, q, t))
                    bal += q
                if rng.random() < 0.25:
                    q = int(rng.integers(1, 10)) * 50
                    out.append((cust, str(sym), d, m.BUY, q, f"11:{int(rng.integers(0, 59)):02d}:00"))
                    bal += q
    return out


def _source_events(schedule, calendar, prices, rng) -> pd.DataFrame:
    close = prices.set_index(["symbol", "date"])["close"]
    rows = []
    for item in schedule:
        cust, sym, d, etype, qty, time = item[:6]
        override = item[6] if len(item) > 6 else None
        deleted = item[7] if len(item) > 7 else False
        day = calendar.days[d]
        if override is not None:
            price = float(override)
        elif etype in (m.BUY, m.SELL):
            price = float(round(close[(sym, day)] * (1 + rng.uniform(-0.004, 0.004))))
        else:
            price = 0.0     # transfers are priced at the close by the pipeline
        rows.append(dict(customer_id=cust, symbol=sym, date=day, time=time, event_type=etype,
                         quantity=int(qty), price=price, is_deleted=int(bool(deleted))))
    df = pd.DataFrame(rows).sort_values(["date", "customer_id", "time"], kind="stable")
    df["source_ref"] = [f"T{i:05d}" for i in range(1, len(df) + 1)]
    return df.reset_index(drop=True)


def build_world(seed: int = 7) -> World:
    rng = np.random.default_rng(seed)
    calendar = m.Calendar(m.business_days(START, N_DAYS))
    symbols = pd.DataFrame([dict(symbol=s, isin=m.make_isin(b), name=n, instrument_class=k,
                                 parent_symbol=par) for s, b, n, k, par, *_ in SYMBOLS])
    delta_isin = symbols.set_index("symbol").loc["DELTA", "isin"]
    lineage = pd.DataFrame([dict(old_isin=m.make_isin(DELTA_OLD_BODY), new_isin=delta_isin,
                                 effective_date=calendar.days[20])])
    fees = pd.DataFrame(FEES, columns=["instrument_class", "sell_fee", "daily_limit"])
    prices = _prices(calendar, rng, FEES)
    day = lambda i: None if i is None else calendar.days[i]     # noqa: E731
    cas = pd.DataFrame([dict(ca_id=c, symbol=s, action_type=t, session_date=day(sd),
                             record_date=day(rd), activation_date=day(ad), period_end=day(pe),
                             value=v, rights_symbol=r)
                        for c, s, t, sd, rd, ad, pe, v, r in CORPORATE_ACTIONS])
    schedule = SCHEDULE + _random_schedule(rng)
    src = _source_events(schedule, calendar, prices, rng)
    # random holders exercise half of their rights on day 36 (needs the rights balance first)
    core = p.build_chain2(src, cas, symbols, prices, calendar)
    d35 = p.holdings_asof(core.daily, calendar.days[35])
    extra = []
    for r in d35[(d35["symbol"] == "GAMMA_R") & (d35["quantity"] > 1)].itertuples():
        if r.customer_id in RANDOM_CUSTOMERS and rng.random() < 0.6:
            extra.append((r.customer_id, "GAMMA_R", 36, m.RIGHTS_EXERCISE, int(r.quantity) // 2,
                          "10:30:00"))
    if extra:
        src = _source_events(schedule + extra, calendar, prices, np.random.default_rng(seed + 1))
    first = src.groupby("customer_id")["date"].min()
    customers = pd.DataFrame({"customer_id": first.index, "opened_on": first.values})
    return World(calendar=calendar, customers=customers, symbols=symbols, lineage=lineage,
                 fees=fees, prices=prices, corporate_actions=cas, source_events=src, rng=rng)


def build_feed(w: World) -> pd.DataFrame:
    """Depository truth: end-of-day share counts on every day the holding changed."""
    core = p.build_chain2(w.source_events, w.corporate_actions, w.symbols, w.prices, w.calendar)
    isin = dict(zip(w.symbols["symbol"], w.symbols["isin"]))
    old_delta = w.lineage["old_isin"].iloc[0]
    move_day = w.lineage["effective_date"].iloc[0]
    ev = core.events[core.events["is_deleted"] == 0]
    custodian_days = set(zip(ev.loc[ev["event_type"].isin([m.CUSTODIAN_IN, m.CUSTODIAN_OUT]),
                                    "customer_id"], ev["symbol"], ev["date"]))
    rows = []
    for (cust, sym), g in core.daily.groupby(["customer_id", "symbol"]):
        prev = None
        for r in g.sort_values("date").itertuples():
            q = max(0, int(r.quantity))
            if q == prev or (prev is None and q == 0):
                continue
            code = isin[sym]
            if sym == "DELTA" and r.date < move_day:
                code = old_delta
            if cust == "C008":
                code = variant_isin(code)
            origin = m.FEED_CUSTODIAN if (cust, sym, r.date) in custodian_days else m.FEED_SETTLEMENT
            rows.append(dict(customer_id=cust, isin=code, date=r.date, quantity=q, origin=origin))
            prev = q
    # depository artefacts: a spurious leave-and-return, and a same-day ISIN change
    rows += [dict(customer_id="C005", isin=isin["DELTA"], date=w.day(22), quantity=0,
                  origin=m.FEED_CUSTODIAN),
             dict(customer_id="C005", isin=isin["DELTA"], date=w.day(23), quantity=600,
                  origin=m.FEED_CUSTODIAN),
             dict(customer_id="C007", isin=variant_isin(isin["BETA"]), date=w.day(12), quantity=0,
                  origin=m.FEED_SETTLEMENT),
             dict(customer_id="C007", isin=isin["BETA"], date=w.day(12), quantity=800,
                  origin=m.FEED_SETTLEMENT)]
    feed = pd.DataFrame(rows).sort_values(["date", "customer_id", "isin"], kind="stable")
    feed.insert(0, "feed_id", [f"F{i:05d}" for i in range(1, len(feed) + 1)])
    return feed.reset_index(drop=True)


def build_cash(w: World) -> pd.DataFrame:
    ev = w.events[w.events["is_deleted"] == 0]
    rows = []
    for cust, g in ev[ev["origin"] == m.ORIGIN_SOURCE].groupby("customer_id"):
        buys = g[g["event_type"].isin([m.BUY, m.OPTION_SETTLE_IN])]
        need = float((buys["quantity"] * buys["price"]).sum())
        first = g["date"].min()
        rows.append(dict(customer_id=cust, date=first, amount=float(np.ceil(need * 1.1 / 1e6) * 1e6),
                         kind="DEPOSIT"))
        if cust in RANDOM_CUSTOMERS[::3]:
            rows.append(dict(customer_id=cust, date=w.calendar.shift(first, 25),
                             amount=-float(np.ceil(need * 0.1 / 1e6) * 1e6), kind="WITHDRAWAL"))
    for r in ev[ev["event_type"] == m.CASH_DIVIDEND].itertuples():
        rows.append(dict(customer_id=r.customer_id, date=w.calendar.shift(r.date, 10),
                         amount=float(r.cash_amount), kind="DIVIDEND_CREDIT"))
    for r in ev[ev["event_type"] == m.RIGHTS_EXERCISE].itertuples():
        rows.append(dict(customer_id=r.customer_id, date=r.date,
                         amount=-float(r.quantity * m.SUBSCRIPTION_PRICE), kind="RIGHTS_PAYMENT"))
    rows += [dict(customer_id=RANDOM_CUSTOMERS[0], date=w.day(10), amount=50_000_000.0,
                  kind="CREDIT_DRAWDOWN"),
             dict(customer_id=RANDOM_CUSTOMERS[0], date=w.day(30), amount=-50_000_000.0,
                  kind="CREDIT_REPAYMENT")]
    cash = pd.DataFrame(rows).sort_values(["date", "customer_id", "kind"], kind="stable")
    cash.insert(0, "movement_id", [f"M{i:04d}" for i in range(1, len(cash) + 1)])
    return cash.reset_index(drop=True)


def resolve_defects(defects) -> list[str]:
    """Accept ``"all"``, ``"none"``, a comma-separated string or a list of names."""
    if defects is None or defects == "none" or defects == []:
        return []
    if defects == "all":
        return list(DEFECTS)
    names = [d.strip() for d in defects.split(",")] if isinstance(defects, str) else list(defects)
    unknown = [d for d in names if d not in DEFECTS]
    if unknown:
        raise ValueError(f"unknown defect(s): {unknown}. Known: {sorted(DEFECTS)}")
    return names


def make_dataset(seed: int = 7, defects="none") -> Dataset:
    """Build the synthetic dataset and inject the requested defects."""
    names = resolve_defects(defects)
    w = build_world(seed)
    manifest = []

    def run(stage):
        for n in names:
            spec = DEFECTS[n]
            if spec.stage == stage:
                target = spec.apply(w)
                manifest.append(dict(name=n, stage=stage, target=target,
                                     description=spec.description,
                                     expected_checks=",".join(sorted(spec.expected_checks)),
                                     benign=str(spec.benign)))

    run("source")
    w.feed = build_feed(w)
    run("feed")
    ch1 = p.build_chain1(w.feed, w.symbols, w.lineage)
    w.chain1_daily, w.chain1_latest = ch1.daily, ch1.latest
    run("trade_feed")
    ch2 = p.build_chain2(w.source_events, w.corporate_actions, w.symbols, w.prices, w.calendar,
                         depository_daily=w.chain1_daily)
    w.events, w.daily, w.latest = ch2.events, ch2.daily, ch2.latest
    run("stored")
    w.cash_movements = build_cash(w)
    w.return_outputs = p.build_return_outputs(w.customers, w.daily, w.prices, w.calendar)
    ext = w.cash_movements[w.cash_movements["kind"].isin(m.EXTERNAL_FLOW_KINDS)]
    w.return_cash_flows = ext[["customer_id", "date", "amount"]].reset_index(drop=True)
    w.app_display = p.build_app_display(w.latest, w.symbols, w.fees, w.prices, w.calendar.as_of)
    run("downstream")

    tables = {
        "calendar": pd.DataFrame({"date": w.calendar.days}),
        "customers": w.customers, "symbols": w.symbols, "isin_lineage": w.lineage,
        "fee_schedule": w.fees, "prices": w.prices, "corporate_actions": w.corporate_actions,
        "depository_feed": w.feed, "depository_daily": w.chain1_daily,
        "depository_latest": w.chain1_latest, "position_events": w.events,
        "positions_daily": w.daily, "positions_latest": w.latest,
        "return_outputs": w.return_outputs, "cash_movements": w.cash_movements,
        "return_cash_flows": w.return_cash_flows, "app_display": w.app_display,
    }
    tables = {k: v.reset_index(drop=True) for k, v in tables.items()}
    return Dataset(tables, manifest)
