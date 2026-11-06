import pandas as pd
import pytest

from portfolio_dq import model as m
from portfolio_dq import pipeline as p

DAYS = m.business_days("2026-04-04", 12)
CAL = m.Calendar(DAYS)
SYMBOLS = pd.DataFrame([dict(symbol="SYM_A", isin=m.make_isin("GB00SYMA000"), name="A",
                             instrument_class=m.EQUITY, parent_symbol=None)])
PRICES = pd.DataFrame(dict(symbol="SYM_A", date=DAYS, close=1000.0))
NO_CA = pd.DataFrame(columns=["ca_id", "symbol", "action_type", "session_date", "record_date",
                              "activation_date", "period_end", "value", "rights_symbol"])


def src(*rows):
    return pd.DataFrame([dict(customer_id="U001", symbol="SYM_A", date=DAYS[d], time=t,
                              event_type=e, quantity=q, price=px, is_deleted=dl,
                              source_ref=f"T{i}")
                         for i, (d, t, e, q, px, dl) in enumerate(rows)])


def test_average_cost_and_realized_pnl_by_hand():
    s = src((0, "09:00:00", m.BUY, 100, 1000, 0), (1, "09:00:00", m.BUY, 100, 2000, 0),
            (2, "09:00:00", m.SELL, 50, 3000, 0))
    ch = p.build_chain2(s, NO_CA, SYMBOLS, PRICES, CAL)
    last = ch.latest.iloc[0]
    assert last["quantity"] == 150
    assert last["avg_cost"] == 1500                 # (100 x 1000 + 100 x 2000) / 200
    assert last["realized_pnl"] == 75_000           # (3000 - 1500) x 50; a sale keeps avg cost
    assert list(ch.events["balance_after"]) == [100, 200, 150]


def test_soft_deleted_row_is_stored_but_not_applied():
    s = src((0, "09:00:00", m.BUY, 100, 1000, 0), (1, "09:00:00", m.BUY, 999, 1000, 1))
    ch = p.build_chain2(s, NO_CA, SYMBOLS, PRICES, CAL)
    assert ch.latest.iloc[0]["quantity"] == 100
    assert ch.events["is_deleted"].sum() == 1


def test_gap_plug_unless_settlement_lag_explains_it():
    s = src((0, "09:00:00", m.BUY, 100, 1000, 0))
    dep = pd.DataFrame(dict(customer_id="U001", symbol="SYM_A", date=[DAYS[1], DAYS[3]],
                            quantity=[100, 130], received_isin=SYMBOLS["isin"][0],
                            row_type=m.ROW_POSITION, registration_type=m.REG_DEPOSITORY))
    ch = p.build_chain2(s, NO_CA, SYMBOLS, PRICES, CAL, depository_daily=dep)
    plugs = ch.events[ch.events["event_type"] == m.UNEXPLAINED_INCREASE]
    assert list(plugs["quantity"]) == [30]           # day 1 agrees; day 3 is a real gap
    assert plugs["date"].iloc[0] == DAYS[3]


def test_dividend_and_bonus_entitlement_from_previous_close():
    cas = pd.DataFrame([dict(ca_id="X1", symbol="SYM_A", action_type=m.DIVIDEND,
                             session_date=DAYS[1], record_date=DAYS[3], activation_date=None,
                             period_end=None, value=50.0, rights_symbol=None),
                        dict(ca_id="X2", symbol="SYM_A", action_type=m.BONUS,
                             session_date=DAYS[2], record_date=DAYS[5], activation_date=DAYS[8],
                             period_end=None, value=0.3, rights_symbol=None)])
    # 101 shares, then 10 more bought ON the dividend record date (not entitled to it)
    s = src((0, "09:00:00", m.BUY, 101, 1000, 0), (3, "10:00:00", m.BUY, 10, 1000, 0))
    ev = p.build_chain2(s, cas, SYMBOLS, PRICES, CAL).events
    div = ev[ev["event_type"] == m.CASH_DIVIDEND].iloc[0]
    assert div["cash_amount"] == 101 * 50 and div["time"] == p.CA_TIME
    grant = ev[ev["event_type"] == m.BONUS_GRANT].iloc[0]
    assert grant["quantity"] == 33                   # floor(111 x 0.3)
    assert ev[ev["event_type"] == m.BONUS_ACTIVATE]["balance_after"].iloc[0] == 144


def feed(rows):
    return pd.DataFrame([dict(feed_id=f"F{i}", customer_id="U001", isin=isin, date=DAYS[d],
                              quantity=q, origin=o) for i, (isin, d, q, o) in enumerate(rows)])


def test_chain1_same_day_isin_change_keeps_largest_row():
    good = SYMBOLS["isin"][0]
    variant = good[:11] + str((int(good[11]) + 1) % 10)
    ch = p.build_chain1(feed([(variant, 2, 0, m.FEED_SETTLEMENT),
                              (good, 2, 500, m.FEED_SETTLEMENT)]), SYMBOLS)
    row = ch.daily.iloc[0]
    assert len(ch.daily) == 1 and row["quantity"] == 500
    assert row["row_type"] == m.ROW_CHANGE_MARKET


def test_chain1_leave_and_return_is_ignored():
    isin = SYMBOLS["isin"][0]
    ch = p.build_chain1(feed([(isin, 0, 600, m.FEED_SETTLEMENT), (isin, 4, 0, m.FEED_CUSTODIAN),
                              (isin, 5, 600, m.FEED_CUSTODIAN)]), SYMBOLS)
    assert list(ch.daily["row_type"]) == [m.ROW_POSITION, m.ROW_IGNORE, m.ROW_JOIN]
    assert ch.latest.iloc[0]["quantity"] == 600


def test_chain1_resolves_old_isin_through_lineage_or_rejects_it():
    old = m.make_isin("IRO3SYMA000")
    lineage = pd.DataFrame([dict(old_isin=old, new_isin=SYMBOLS["isin"][0],
                                 effective_date=DAYS[3])])
    f = feed([(old, 1, 100, m.FEED_SETTLEMENT)])
    assert p.build_chain1(f, SYMBOLS, lineage).daily.iloc[0]["symbol"] == "SYM_A"
    assert len(p.build_chain1(f, SYMBOLS).rejected) == 1


def test_app_display_formula_chain_by_hand():
    latest = pd.DataFrame([dict(customer_id="U001", symbol="SYM_A", quantity=100,
                                avg_cost=800.0)])
    fees = pd.DataFrame([dict(instrument_class=m.EQUITY, sell_fee=0.01, daily_limit=0.05)])
    row = p.build_app_display(latest, SYMBOLS, fees, PRICES, DAYS[-1]).iloc[0]
    assert row["market_value"] == 99_000             # 100 x 1000 x (1 - 1%)
    assert row["unrealized_pnl"] == 19_000           # 99,000 - 80,000
    assert row["return_pct"] == pytest.approx(0.2375)
    assert row["break_even_price"] == 808            # 800 x 1.01
