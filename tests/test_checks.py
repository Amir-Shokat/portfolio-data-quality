import pytest

from portfolio_dq.checks import REGISTRY, catalogue, run_all, run_check
from portfolio_dq.generator import DEFECTS, make_dataset
from portfolio_dq.model import SCOPE_LABELS


def fired(ds):
    results = run_all(ds)
    assert not [r.spec.id for r in results if r.status == "ERROR"]
    return {r.spec.id for r in results if r.fired}


def test_catalogue_is_complete_and_labelled():
    cat = catalogue()
    assert cat["id"].is_unique and len(cat) == len(REGISTRY) >= 25
    assert set(cat["family"]) == {"A", "B", "C", "D"}
    assert set(cat["scope"]) <= set(SCOPE_LABELS.values())
    assert {"WITHIN-CHAIN (depository route)", "WITHIN-CHAIN (transactional route)",
            "CROSS-CHAIN"} <= set(cat["scope"])
    assert cat["severity"].isin(["critical", "high", "medium", "low", "info"]).all()
    assert cat["owner"].str.len().gt(0).all()
    assert cat["scope"].iloc[0] == "VALIDITY GATE"          # gates run first


def test_clean_data_passes_every_check(clean):
    assert fired(clean) == set()


@pytest.mark.parametrize("name", list(DEFECTS))
def test_each_defect_fires_exactly_its_checks(name):
    ds = make_dataset(seed=7, defects=name)
    assert fired(ds) == set(DEFECTS[name].expected_checks)


def test_all_defects_together(broken):
    expected = set().union(*(d.expected_checks for d in DEFECTS.values()))
    assert fired(broken) == expected


def test_balance_break_is_localised(broken):
    e = run_check("A5B", broken).exceptions
    assert list(e["diff"]) == [100, -100]                   # wrong row, then the row after it
    assert set(e["customer_id"]) == {"C022"}


def test_missing_dividend_value_weighted(broken):
    e = run_check("A12", broken).exceptions
    row = e.iloc[0]
    assert (row["customer_id"], row["mismatch_type"]) == ("C024", "MISSING_CA_EVENT")
    assert row["expected"] == 600 * 600.0                   # 600 shares x 600 IRR per share


def test_rollup_reports_both_assertions(broken):
    e = run_check("A4", broken).exceptions
    assert set(e["mismatch_type"]) == {"TERMINAL_BALANCE_BREAK", "ROLLUP_BREAK"}
    assert set(e["diff"]) == {-50, 50}


def test_negative_ledger_balance_is_masked_by_rollup(broken):
    r = run_check("A0", broken)
    neg = r.exceptions[r.exceptions["mismatch_type"] == "NEGATIVE_BALANCE"]
    assert set(neg["source"]) == {"position_events"}        # never reaches the daily table
    assert neg["value"].min() == -700
    assert any("masking" in n for n in r.notes)


def test_plugs_split_by_direction(broken):
    r = run_check("A1", broken)
    assert r.metrics["increases"] == 2 and r.metrics["decreases"] == 0
    assert sorted(r.exceptions["diff"]) == [500, 700]


def test_settlement_lag_is_not_an_exception():
    ds = make_dataset(seed=7, defects="settlement_lag")
    r = run_check("A2X_HIST", ds)
    assert r.n_exceptions == 0 and r.metrics["settlement_lag_days"] > 0


def test_fragmentation_counts_variants_but_reports_split(broken):
    r = run_check("A11", broken)
    assert set(r.exceptions["mismatch_type"]) == {"FRAGMENTED_SECURITY", "SPLIT_ACROSS_SYMBOLS"}
    assert r.metrics["check_digit_variants"] > 0            # known variants: counted only


def test_summary_row_has_rate_and_customer_count(broken):
    row = run_check("A8", broken).summary_row()
    assert row["scope"] == "WITHIN-CHAIN (depository route)"
    assert row["customers"] > 0 and 0 < row["rate"] < 1


def test_broken_check_is_reported_not_raised(clean):
    bad = type(clean)({k: v.iloc[0:0] for k, v in clean.tables.items()
                       if k != "position_events"})
    bad.tables["position_events"] = clean["position_events"].drop(columns=["balance_after"])
    assert run_check("A5B", bad).status == "ERROR"
