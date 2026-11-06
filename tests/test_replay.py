import pytest

from portfolio_dq.replay import ERROR, INFO, rebuild_compare
from portfolio_dq.scenarios import SCENARIOS, judge, run_scenario, run_scenarios
from portfolio_dq.generator import make_dataset
from portfolio_dq.scenarios import SCENARIO_DEFECTS


@pytest.fixture(scope="module")
def scenario_ds():
    return make_dataset(seed=7, defects=SCENARIO_DEFECTS)


def test_clean_rebuild_matches_everything(clean):
    r = rebuild_compare(clean, clean["customers"]["customer_id"])
    c = r.counters
    assert c["units_with_error"] == 0 and c["units_failed"] == 0 and c["units_applied"] == 0
    assert c["total_units"] == c["units_matched"] > 50
    assert r.differences.empty


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario_outcome(scenario_ds, scenario):
    rec = run_scenario(scenario_ds, scenario)
    assert rec.candidate, "the evidence query must return the scenario customer"
    assert rec.outcome in ("CORRECTLY_DETECTED", "NO_DISCREPANCY_AS_EXPECTED")


def test_info_only_unit_counts_as_matched(scenario_ds):
    r = rebuild_compare(scenario_ds, "C027", "BETA")
    assert set(r.differences["severity"]) == {INFO}
    assert set(r.differences["fields"]) == {"registration_type"}
    assert r.counters["units_with_error"] == 0 and r.counters["units_matched"] == 1


def test_missing_rows_are_always_errors(scenario_ds):
    r = rebuild_compare(scenario_ds, "C024", "ALPHA")
    missing = r.differences[r.differences["status"].str.startswith("MISSING")]
    assert len(missing) == 2 and set(missing["severity"]) == {ERROR}
    assert set(missing["fields"]) == {""}                   # no field list for a missing row


def test_tamper_works_on_a_copy(scenario_ds):
    before = scenario_ds["positions_latest"]["quantity"].sum()
    s = next(x for x in SCENARIOS if x.tamper.startswith("UPDATE"))
    run_scenario(scenario_ds, s)
    assert scenario_ds["positions_latest"]["quantity"].sum() == before


def test_wrong_customer_is_not_a_candidate(scenario_ds):
    from dataclasses import replace
    rec = run_scenario(scenario_ds, replace(SCENARIOS[3], customer="C001"))
    assert rec.outcome == "NOT_A_CANDIDATE"


def test_judge_covers_all_outcomes():
    a, b = frozenset({("t", "MISMATCH")}), frozenset({("t", "MISSING_STORED")})
    assert judge(frozenset(), frozenset()) == "NO_DISCREPANCY_AS_EXPECTED"
    assert judge(a, a) == "CORRECTLY_DETECTED"
    assert judge(a, frozenset()) == "MISSED_DISCREPANCY"
    assert judge(frozenset(), a) == "UNEXPECTED_DISCREPANCY"
    assert judge(a, b) == "MISCLASSIFIED"


def test_run_scenarios_subset():
    recs = run_scenarios(ids=["S01", "S11"])
    assert [r.scenario.id for r in recs] == ["S01", "S11"]
    assert all(r.outcome == "NO_DISCREPANCY_AS_EXPECTED" for r in recs)
