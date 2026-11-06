import pandas as pd
import pytest

from portfolio_dq import model as m


def test_isin_check_digit_known_value():
    # the ISO 6166 worked example: US0378331005
    assert m.isin_check_digit("US037833100") == "5"


def test_make_isin_is_clean_and_variant_is_flagged():
    isin = m.make_isin("GB00ALPH000")
    assert m.isin_issues(isin) == []
    variant = isin[:11] + str((int(isin[11]) + 3) % 10)
    assert m.isin_issues(variant) == ["CHECK_DIGIT"]
    assert m.canonical_isin(variant) == m.canonical_isin(isin) == "GB00ALPH000"


def test_isin_structural_issues():
    assert "LENGTH" in m.isin_issues("GB00ALPH00")
    assert "PREFIX" in m.isin_issues(m.make_isin("USO1ALPH000"))
    assert m.isin_issues(None) == ["LENGTH"]


def test_entitlement_floors_and_ignores_non_positive():
    assert m.entitlement(1234, 0.5) == 617
    assert m.entitlement(1001, 0.4) == 400          # 400.4 -> 400
    assert m.entitlement(0, 0.5) == 0
    assert m.entitlement(-100, 0.5) == 0


def test_adjusted_base_price_by_hand():
    assert m.adjusted_base_price(12_000, [(m.DIVIDEND, 600)]) == 11_400
    assert m.adjusted_base_price(9_000, [(m.BONUS, 0.5)]) == pytest.approx(6_000)
    # rights at 40% paid at par 1,000: (15,000 + 400) / 1.4 = 11,000
    assert m.adjusted_base_price(15_000, [(m.RIGHTS, 0.4)]) == pytest.approx(11_000)


def test_calendar_skips_saturday_and_sunday():
    days = m.business_days("2026-04-04", 10)        # 4 April 2026 is a Saturday
    assert not any(d.dayofweek in (5, 6) for d in days)
    cal = m.Calendar(days)
    assert cal.between(days[0], days[5]) == 5
    assert cal.shift(days[4], 1) == days[5]
    assert cal.shift(days[0], -1) is None
    assert cal.index(pd.Timestamp("2026-04-12")) == cal.index(days[4])   # Sunday -> Friday
