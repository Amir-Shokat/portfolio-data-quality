"""Command line: ``portfolio-dq <command>``.

Commands:
  generate    build the synthetic dataset (SQLite file or CSV folder) with chosen defects
  catalogue   print the check catalogue
  check       run the catalogue on a dataset; optionally write the Excel tracker
  replay      rebuild and compare one customer (report-only)
  scenarios   run the reconciliation test protocol; optionally write the Excel tracker
  compress    compress daily positions into ranges, print fingerprints and anomalies
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .checks import catalogue, run_all, summary
from .compression import (anomalies, compression_ratio, dataset_fingerprint, fingerprint,
                          to_ranges)
from .dataset import Dataset
from .generator import DEFECTS, make_dataset
from .replay import rebuild_compare
from .scenarios import records_frame, run_scenarios
from .tracker import write_tracker

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)


def _load(path: str) -> Dataset:
    return Dataset.load(path)


def cmd_generate(a) -> int:
    ds = make_dataset(a.seed, a.defects)
    out = Path(a.out)
    if out.suffix in (".sqlite", ".db"):
        ds.to_sqlite(out)
    else:
        ds.to_csv(out)
    print(f"wrote {out} ({sum(len(t) for t in ds.tables.values())} rows, "
          f"{len(ds.manifest)} injected defects)")
    return 0


def cmd_catalogue(a) -> int:
    print(catalogue()[["id", "family", "scope", "severity", "owner", "title"]].to_string(index=False))
    return 0


def cmd_check(a) -> int:
    ds = _load(a.db)
    ids = a.only.split(",") if a.only else None
    results = run_all(ds, ids=ids, lag_days=a.lag_days)
    s = summary(results)
    cols = ["id", "scope", "status", "exceptions", "customers", "denominator", "magnitude", "worst"]
    print(s[cols].to_string(index=False))
    if a.xlsx:
        print(f"wrote {write_tracker(a.xlsx, results=results)}")
    return 1 if a.fail_on_exception and (s["status"] == "FAIL").any() else 0


def cmd_replay(a) -> int:
    ds = _load(a.db)
    report = rebuild_compare(ds, a.customer, a.symbol.split(",") if a.symbol else None)
    print(report.to_text())
    return 0


def cmd_scenarios(a) -> int:
    records = run_scenarios(seed=a.seed)
    df = records_frame(records)
    print(df[["Test ID", "Scenario", "Customer-symbol", "Actual report", "Outcome"]]
          .to_string(index=False))
    if a.xlsx:
        ds = make_dataset(a.seed, "all")
        print(f"wrote {write_tracker(a.xlsx, results=run_all(ds), records=records)}")
    bad = df["Outcome"].isin(["MISSED_DISCREPANCY", "UNEXPECTED_DISCREPANCY", "MISCLASSIFIED",
                              "EXECUTION_FAILED"])
    return 1 if bad.any() else 0


def cmd_compress(a) -> int:
    ds = _load(a.db)
    if a.table == "depository_daily":
        rows = ds["depository_daily"]
        rows = rows[rows["row_type"] != "IGNORE"]
    else:
        rows = ds["positions_daily"]
    ranges = to_ranges(rows, ds.calendar)
    daily_equivalent = int(ranges["days"].sum())
    print(f"{len(rows)} stored rows -> {len(ranges)} ranges covering {daily_equivalent} "
          f"customer-symbol-days (x{compression_ratio(daily_equivalent, ranges)})")
    fp = fingerprint(ranges)
    print("table fingerprint:", dataset_fingerprint(fp))
    print(fp.head(a.head).to_string())
    an = anomalies(ranges)
    print(f"{len(an)} anomalous ranges")
    if len(an):
        print(an.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="portfolio-dq", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="build the synthetic dataset")
    g.add_argument("--out", required=True, help="file.sqlite or a folder for CSV files")
    g.add_argument("--defects", default="all",
                   help=f"'all', 'none' or a comma list of: {', '.join(DEFECTS)}")
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(fn=cmd_generate)
    c = sub.add_parser("catalogue", help="print the check catalogue")
    c.set_defaults(fn=cmd_catalogue)
    k = sub.add_parser("check", help="run the check catalogue")
    k.add_argument("--db", required=True, help="SQLite file or CSV folder")
    k.add_argument("--only", help="comma list of check ids")
    k.add_argument("--lag-days", type=int, default=2)
    k.add_argument("--xlsx", help="write the tracker workbook here")
    k.add_argument("--fail-on-exception", action="store_true", help="exit 1 if any check fails")
    k.set_defaults(fn=cmd_check)
    r = sub.add_parser("replay", help="rebuild and compare one customer (report-only)")
    r.add_argument("--db", required=True)
    r.add_argument("--customer", required=True)
    r.add_argument("--symbol", help="comma list; default: all symbols of the customer")
    r.set_defaults(fn=cmd_replay)
    s = sub.add_parser("scenarios", help="run the reconciliation test protocol")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--xlsx", help="write the tracker workbook here")
    s.set_defaults(fn=cmd_scenarios)
    z = sub.add_parser("compress", help="compress daily positions into ranges")
    z.add_argument("--db", required=True)
    z.add_argument("--table", choices=["positions_daily", "depository_daily"],
                   default="positions_daily")
    z.add_argument("--head", type=int, default=5)
    z.set_defaults(fn=cmd_compress)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
