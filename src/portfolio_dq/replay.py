"""Report-only "rebuild and compare".

For each selected customer, rebuild both routes in memory from the stored inputs (the raw
depository feed, the source rows of the ledger, corporate actions, prices and the
calendar), then compare the rebuilt tables with the stored ones, row by row.

Nothing is written back. The rebuild is treated as the reference for the comparison; it
is not proof that either side is right.

Row statuses: ``MATCH``, ``MISMATCH`` (both sides have the row, a compared field differs),
``MISSING_STORED`` (rebuilt row absent from storage) and ``MISSING_REBUILT`` (stored row
the rebuild did not produce). Missing rows are always ``ERROR``. A mismatch is ``INFO``
only when every differing field is an informational field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import model as m
from .checks.common import equal
from .pipeline import build_chain1, build_chain2

MATCH, MISMATCH, MISSING_STORED, MISSING_REBUILT = (
    "MATCH", "MISMATCH", "MISSING_STORED", "MISSING_REBUILT")
ERROR, INFO = "ERROR", "INFO"

#: table -> (row key after customer+symbol, {field: severity})
COMPARED = {
    "depository_daily": (["date"], {"quantity": ERROR, "row_type": ERROR,
                                    "received_isin": ERROR, "registration_type": INFO}),
    "depository_latest": ([], {"date": ERROR, "quantity": ERROR, "row_type": ERROR}),
    "position_events": (["date", "time", "event_type", "ref"],
                        {"quantity": ERROR, "balance_after": ERROR, "ca_value": ERROR}),
    "positions_daily": (["date"], {"quantity": ERROR}),
    "positions_latest": ([], {"date": ERROR, "quantity": ERROR}),
}
#: not compared, on purpose: average costs, realized P&L, prices, cash amounts. A clean
#: comparison therefore does not prove every money field is right.
NOT_COMPARED = ("avg_cost", "avg_cost_after", "realized_pnl", "price", "cash_amount")


@dataclass
class ReplayReport:
    units: pd.DataFrame          # one row per customer+symbol: status, error and info counts
    tables: pd.DataFrame         # per table: matched, differing, info, only_stored, only_rebuilt
    differences: pd.DataFrame    # non-matching rows only
    warnings: list = field(default_factory=list)

    @property
    def counters(self) -> dict:
        u = self.units
        return dict(total_units=len(u),
                    units_with_error=int((u["status"] == "ERROR").sum()),
                    units_matched=int(u["status"].isin(["MATCH", "INFO_ONLY"]).sum()),
                    units_failed=int((u["status"] == "FAILED").sum()),
                    units_applied=0)

    def to_text(self) -> str:
        c = self.counters
        lines = ["units: " + ", ".join(f"{k}={v}" for k, v in c.items())]
        if len(self.tables):
            lines.append(self.tables.to_string(index=False))
        if len(self.differences):
            cols = ["customer_id", "symbol", "table", "key", "status", "severity", "fields",
                    "stored", "rebuilt"]
            lines.append(self.differences[cols].fillna("").to_string(index=False))
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def _source_rows(ds, customer: str) -> pd.DataFrame:
    ev = ds["position_events"]
    src = ev[(ev["customer_id"] == customer) & (ev["origin"] == m.ORIGIN_SOURCE)]
    return src[["customer_id", "symbol", "date", "time", "event_type", "quantity", "price",
                "is_deleted", "source_ref"]].reset_index(drop=True)


def rebuild_customer(ds, customer: str) -> dict[str, pd.DataFrame]:
    """Rebuild all five holdings tables for one customer, in memory."""
    feed = ds["depository_feed"]
    ch1 = build_chain1(feed[feed["customer_id"] == customer], ds["symbols"], ds["isin_lineage"])
    src = _source_rows(ds, customer)
    out = {"depository_daily": ch1.daily, "depository_latest": ch1.latest}
    if len(src) or len(ch1.daily):
        ch2 = build_chain2(src, ds["corporate_actions"], ds["symbols"], ds["prices"],
                           ds.calendar, depository_daily=ch1.daily)
        out.update(position_events=ch2.events, positions_daily=ch2.daily,
                   positions_latest=ch2.latest)
    else:
        out.update(position_events=ds["position_events"].iloc[0:0],
                   positions_daily=ds["positions_daily"].iloc[0:0],
                   positions_latest=ds["positions_latest"].iloc[0:0])
    return out


def _prepare(table: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if table == "position_events":
        df = df[df["is_deleted"] == 0]          # soft-deleted rows are ignored on both sides
        df["ref"] = df["source_ref"].where(df["source_ref"].notna(), df["ca_id"]).fillna("")
    return df


def _fmt(v) -> str:
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "(unset)"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def compare_table(table: str, stored: pd.DataFrame, rebuilt: pd.DataFrame,
                  warnings: list | None = None) -> pd.DataFrame:
    """Row-by-row comparison of one table; returns one row per compared row with a status."""
    extra, fields = COMPARED[table]
    key = ["customer_id", "symbol"] + extra
    s, r = _prepare(table, stored), _prepare(table, rebuilt)
    dups = s[s.duplicated(key, keep=False)]
    if len(dups) and warnings is not None:
        warnings.append(f"{table}: {len(dups)} stored rows share a key; the last one is compared")
    s = s.drop_duplicates(key, keep="last").set_index(key)
    r = r.drop_duplicates(key, keep="last").set_index(key)
    rows = []
    for k in s.index.union(r.index):
        k = k if isinstance(k, tuple) else (k,)
        base = dict(customer_id=k[0], symbol=k[1], table=table,
                    key="|".join(_fmt(x) for x in k[2:]) or "-")
        if k not in r.index:
            rows.append(dict(base, status=MISSING_REBUILT, severity=ERROR, fields=""))
            continue
        if k not in s.index:
            rows.append(dict(base, status=MISSING_STORED, severity=ERROR, fields=""))
            continue
        a, b = s.loc[k], r.loc[k]
        diff = [f for f in fields if not equal(a[f], b[f])]
        if not diff:
            rows.append(dict(base, status=MATCH, severity="", fields=""))
            continue
        sev = ERROR if any(fields[f] == ERROR for f in diff) else INFO
        rows.append(dict(base, status=MISMATCH, severity=sev, fields=",".join(diff),
                         stored=";".join(_fmt(a[f]) for f in diff),
                         rebuilt=";".join(_fmt(b[f]) for f in diff)))
    cols = ["customer_id", "symbol", "table", "key", "status", "severity", "fields", "stored",
            "rebuilt"]
    return pd.DataFrame(rows, columns=cols)


def rebuild_compare(ds, customers, symbols=None) -> ReplayReport:
    """Rebuild and compare the selected customers (optionally limited to some symbols)."""
    customers = [customers] if isinstance(customers, str) else list(customers)
    symbols = None if symbols is None else ([symbols] if isinstance(symbols, str) else list(symbols))
    compared, warnings, failed = [], [], []
    for cust in customers:
        try:
            rebuilt = rebuild_customer(ds, cust)
            for table in COMPARED:
                st = ds[table][ds[table]["customer_id"] == cust]
                rb = rebuilt[table]
                if symbols is not None:
                    st, rb = st[st["symbol"].isin(symbols)], rb[rb["symbol"].isin(symbols)]
                compared.append(compare_table(table, st, rb, warnings))
        except Exception as exc:          # one failing unit must not stop the others
            failed.append(dict(customer_id=cust, symbol="*", status="FAILED", errors=0, infos=0,
                               note=f"{exc.__class__.__name__}: {exc}"))
    allrows = pd.concat(compared, ignore_index=True) if compared else compare_table(
        "positions_latest", ds["positions_latest"].iloc[0:0], ds["positions_latest"].iloc[0:0])
    units = []
    for (c, s), g in allrows.groupby(["customer_id", "symbol"]):
        errors = int((g["severity"] == ERROR).sum())
        infos = int((g["severity"] == INFO).sum())
        units.append(dict(customer_id=c, symbol=s, errors=errors, infos=infos, note="",
                          status="ERROR" if errors else "INFO_ONLY" if infos else "MATCH"))
    units = pd.DataFrame(units + failed,
                         columns=["customer_id", "symbol", "status", "errors", "infos", "note"])
    tables = []
    for t in COMPARED:
        g = allrows[allrows["table"] == t]
        tables.append(dict(table=t, matched=int((g["status"] == MATCH).sum()),
                           differing=int((g["status"] == MISMATCH).sum()),
                           info=int((g["severity"] == INFO).sum()),
                           only_stored=int((g["status"] == MISSING_REBUILT).sum()),
                           only_rebuilt=int((g["status"] == MISSING_STORED).sum())))
    diffs = allrows[allrows["status"] != MATCH].reset_index(drop=True)
    return ReplayReport(units=units, tables=pd.DataFrame(tables), differences=diffs,
                        warnings=warnings)
