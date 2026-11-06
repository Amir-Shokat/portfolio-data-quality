"""Helpers shared by the checks: filtered views, as-of series and price lookups."""

from __future__ import annotations

import bisect
from collections import defaultdict

import pandas as pd

from .. import model as m


def live_events(ds) -> pd.DataFrame:
    """Ledger rows that are not soft-deleted. Every check uses this, never the raw table."""
    ev = ds["position_events"]
    return ev[ev["is_deleted"] == 0]


def ordered(ev: pd.DataFrame) -> pd.DataFrame:
    return ev.sort_values(["customer_id", "symbol", "date", "time", "event_id"], kind="stable")


def dedupe_daily(ds) -> pd.DataFrame:
    """positions_daily without exact duplicate rows (A9 reports those separately)."""
    return ds["positions_daily"].drop_duplicates()


def with_canonical(ds, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["canonical"] = out["symbol"].map(ds.canonical_map()).fillna(out["symbol"])
    return out


class Prices:
    """Last close on or before a day, per symbol."""

    def __init__(self, prices: pd.DataFrame):
        self._s = {}
        for sym, g in prices.sort_values("date").groupby("symbol"):
            self._s[sym] = (list(g["date"]), list(g["close"]))

    def close(self, symbol, day):
        dates, closes = self._s.get(symbol, ([], []))
        i = bisect.bisect_right(dates, pd.Timestamp(day)) - 1
        return closes[i] if i >= 0 else None

    def last_date(self, symbol, day):
        dates, _ = self._s.get(symbol, ([], []))
        i = bisect.bisect_right(dates, pd.Timestamp(day)) - 1
        return dates[i] if i >= 0 else None


class Series:
    """Quantity as of any day for one unit, from rows on change days."""

    def __init__(self, points: list[tuple]):
        points = sorted(points)
        self.dates = [d for d, _ in points]
        self.values = [q for _, q in points]

    def at(self, day) -> float:
        if day is None:
            return float("nan")
        i = bisect.bisect_right(self.dates, pd.Timestamp(day)) - 1
        return float(self.values[i]) if i >= 0 else 0.0

    def change_dates(self) -> list:
        out, prev = [], 0.0
        for d, q in zip(self.dates, self.values):
            if q != prev:
                out.append(d)
            prev = q
        return out


def unit_series(rows: pd.DataFrame, key=("customer_id", "canonical")) -> dict[tuple, Series]:
    """Sum the as-of quantities of every symbol in a unit (e.g. all symbols sharing one
    canonical ISIN) and return one series per unit."""
    per_symbol = defaultdict(list)
    for r in rows.itertuples(index=False):
        k = tuple(getattr(r, c) for c in key)
        per_symbol[(k, r.symbol)].append((r.date, float(r.quantity)))
    grouped = defaultdict(dict)
    for (k, sym), pts in per_symbol.items():
        # one value per day: the last row of the day wins
        grouped[k][sym] = Series(list({d: q for d, q in sorted(pts)}.items()))
    out = {}
    for k, syms in grouped.items():
        dates = sorted({d for s in syms.values() for d in s.dates})
        out[k] = Series([(d, sum(s.at(d) for s in syms.values())) for d in dates])
    return out


def chain1_rows(ds) -> pd.DataFrame:
    d = ds["depository_daily"]
    return with_canonical(ds, d[d["row_type"] != m.ROW_IGNORE])


def chain2_rows(ds) -> pd.DataFrame:
    return with_canonical(ds, dedupe_daily(ds))


def lag_explained(dep: Series, target: float, day, calendar: m.Calendar, lag_days: int) -> bool:
    """True if the depository reaches ``target`` within ``lag_days`` trading days after ``day``."""
    return any(dep.at(calendar.shift(day, k)) == target for k in range(1, lag_days + 1)
               if calendar.shift(day, k) is not None)


def equal(a, b, tol: float = 1e-9) -> bool:
    """Null-safe equality with a numeric tolerance."""
    a_na, b_na = pd.isna(a), pd.isna(b)
    if a_na or b_na:
        return bool(a_na and b_na)
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return str(a) == str(b)
