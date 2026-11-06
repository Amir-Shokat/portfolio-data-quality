from openpyxl import load_workbook

from portfolio_dq.checks import run_all
from portfolio_dq.cli import main
from portfolio_dq.dataset import Dataset
from portfolio_dq.scenarios import run_scenarios
from portfolio_dq.tracker import exception_register, write_tracker


def test_sqlite_and_csv_round_trip(broken, tmp_path):
    for target in (tmp_path / "x.sqlite", tmp_path / "csv"):
        if target.suffix:
            broken.to_sqlite(target)
        else:
            broken.to_csv(target)
        back = Dataset.load(target)
        assert len(back["position_events"]) == len(broken["position_events"])
        assert len(back.manifest) == len(broken.manifest)
        assert {r.spec.id for r in run_all(back) if r.fired} == \
            {r.spec.id for r in run_all(broken) if r.fired}


def test_sql_access_returns_dates(broken):
    df = broken.sql("SELECT customer_id, date FROM positions_daily WHERE quantity < ?", (0,))
    assert df.empty
    df = broken.sql("SELECT MAX(date) AS date FROM calendar")
    assert df["date"].iloc[0] == broken.as_of        # date columns come back as Timestamps


def test_tracker_workbook(broken, tmp_path):
    results = run_all(broken)
    records = run_scenarios(ids=["S01", "S04", "S14"])
    path = write_tracker(tmp_path / "tracker.xlsx", results=results, records=records)
    wb = load_workbook(path)
    assert wb.sheetnames == ["Summary", "Exceptions", "Catalogue", "Scenario tests",
                             "Replay differences"]
    tests = wb["Scenario tests"]
    header = [c.value for c in tests[1]]
    assert "Outcome" in header and tests.max_row == 4
    assert tests.data_validations.dataValidation          # outcome drop-down
    assert wb["Exceptions"].max_row - 1 == len(exception_register(results))


def test_cli_generate_and_check(tmp_path, capsys):
    out = tmp_path / "data"
    assert main(["generate", "--out", str(out), "--defects", "avg_cost_jump"]) == 0
    assert main(["check", "--db", str(out), "--only", "A6,A5B", "--fail-on-exception"]) == 1
    text = capsys.readouterr().out
    assert "A6" in text and "FAIL" in text and "PASS" in text


def test_cli_replay(tmp_path, capsys):
    out = tmp_path / "d.sqlite"
    main(["generate", "--out", str(out), "--defects", "missing_dividend_event"])
    assert main(["replay", "--db", str(out), "--customer", "C024", "--symbol", "ALPHA"]) == 0
    assert "MISSING_STORED" in capsys.readouterr().out
