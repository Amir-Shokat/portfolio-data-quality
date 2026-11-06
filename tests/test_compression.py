import numpy as np
import pandas as pd
import pytest

from portfolio_dq import model as m
from portfolio_dq.checks import run_check
from portfolio_dq.compression import (anomalies, compare_fingerprints, dataset_fingerprint,
                                      expand_ranges, fingerprint, to_ranges)
from portfolio_dq.masking import compress_magnitudes, customer_multiplier, mask_ledger

DAYS = m.business_days("2026-04-04", 10)
CAL = m.Calendar(DAYS)


def rows(points, cust="U001", sym="SYM_A"):
    return pd.DataFrame([dict(customer_id=cust, symbol=sym, date=DAYS[d], quantity=q)
                         for d, q in points])


def test_ranges_by_hand_and_round_trip():
    r = to_ranges(rows([(0, 100), (2, 100), (4, 150), (7, 0)]), CAL)
    assert list(r["quantity"]) == [100, 150, 0]          # day 2 repeats day 0: merged
    assert list(r["days"]) == [4, 3, 3]
    assert r["end"].iloc[-1] == CAL.as_of
    daily = expand_ranges(r, CAL)
    assert len(daily) == 10 and daily["quantity"].tolist()[3:5] == [100, 150]


def test_fingerprint_changes_only_when_history_changes():
    a = fingerprint(to_ranges(rows([(0, 100), (4, 150)]), CAL))
    same = fingerprint(to_ranges(rows([(0, 100), (2, 100), (4, 150)]), CAL))
    moved = fingerprint(to_ranges(rows([(0, 100), (5, 150)]), CAL))
    assert a.equals(same)
    assert dataset_fingerprint(a) == dataset_fingerprint(same) != dataset_fingerprint(moved)
    diff = compare_fingerprints(a, moved)
    assert list(diff["status"]) == ["DIFFERENT"]


def test_anomalies_negative_and_flicker():
    r = to_ranges(pd.concat([rows([(0, 600), (3, 0), (4, 600)]),
                             rows([(0, 100), (6, -50)], cust="U002")]), CAL)
    a = anomalies(r)
    assert set(zip(a["customer_id"], a["anomaly"])) == {("U001", "FLICKER"),
                                                         ("U002", "NEGATIVE_HOLDING")}


def test_depository_flicker_found_in_example(broken):
    d = broken["depository_daily"]
    a = anomalies(to_ranges(d[d["row_type"] != m.ROW_IGNORE], broken.calendar))
    assert ("C005", "FLICKER") in set(zip(a["customer_id"], a["anomaly"]))


def test_multiplier_is_deterministic_and_bounded():
    ks = [customer_multiplier(f"C{i:03d}") for i in range(500)]
    assert all(0.5 <= k <= 2.0 for k in ks)
    assert customer_multiplier("C001") == customer_multiplier("C001")
    assert len(set(ks)) > 300                          # not unique, but well spread


def test_masked_ledger_still_passes_continuity(clean):
    masked = clean["position_events"].pipe(mask_ledger)
    ds = type(clean)(dict(clean.tables, position_events=masked))
    assert run_check("A5B", ds).n_exceptions == 0
    assert not masked["quantity"].equals(clean["position_events"]["quantity"])


def test_rounding_each_number_separately_breaks_the_identity():
    q = pd.Series([333, 333, 334])                     # sums to 1000
    k = 1.0015
    assert (q * k).round().sum() != round(1000 * k)    # why balances are recomputed


def test_compress_magnitudes_keeps_zero_sign_and_order():
    x = pd.Series([0, 10, 100, 1_000, 10_000, 100_000, -5_000], dtype=float)
    y = compress_magnitudes(x, target_ratio=5.0, p_low=0, p_high=100)
    assert y[0] == 0 and y.iloc[-1] < 0
    pos = y[x > 0]
    assert pos.is_monotonic_increasing
    assert pos.max() / pos.min() == pytest.approx(5.0, rel=1e-6)


def test_compress_magnitudes_leaves_tight_groups_alone():
    x = pd.Series([100.0, 120, 150])
    assert compress_magnitudes(x).equals(x)
    assert np.isnan(compress_magnitudes(pd.Series([np.nan, 5.0]))[0])
