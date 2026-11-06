"""Compress daily position histories into ranges, fingerprint them, and flag anomalies.

A daily position series is mostly repetition: a customer holds the same quantity for weeks.
Storing one row per *range* of equal quantity keeps all the information in far fewer rows,
and a hash of the ranges (a fingerprint) lets two derivations of the same history be
compared unit by unit without moving the rows themselves. Anomalies such as negative
holdings or a one-day dip to zero stand out in the range form.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from .model import Calendar

RANGE_COLUMNS = ["customer_id", "symbol", "start", "end", "quantity", "days"]


def to_ranges(rows: pd.DataFrame, calendar: Calendar, value: str = "quantity",
              key=("customer_id", "symbol")) -> pd.DataFrame:
    """Collapse rows on change days (or on every day) into ranges of equal ``value``.

    A range runs from the day the value took effect to the trading day before the next
    change, or to the as-of date for the last range. ``days`` counts trading days.
    """
    key = list(key)
    out = []
    for k, g in rows.sort_values("date").groupby(key, sort=True):
        k = k if isinstance(k, tuple) else (k,)
        g = g.drop_duplicates("date", keep="last")
        points = list(zip(g["date"], g[value]))
        merged = []
        for d, q in points:
            if merged and merged[-1][1] == q:
                continue
            merged.append((d, q))
        for i, (d, q) in enumerate(merged):
            end = calendar.shift(merged[i + 1][0], -1) if i + 1 < len(merged) else calendar.as_of
            out.append(dict(zip(key, k), start=d, end=end, quantity=q,
                            days=calendar.between(d, end) + 1))
    return pd.DataFrame(out, columns=key + ["start", "end", "quantity", "days"])


def expand_ranges(ranges: pd.DataFrame, calendar: Calendar,
                  key=("customer_id", "symbol")) -> pd.DataFrame:
    """Inverse of :func:`to_ranges`: one row per unit and trading day."""
    key = list(key)
    rows = []
    for r in ranges.itertuples(index=False):
        for d in calendar:
            if r.start <= d <= r.end:
                rows.append(dict({c: getattr(r, c) for c in key}, date=d, quantity=r.quantity))
    return pd.DataFrame(rows, columns=key + ["date", "quantity"])


def fingerprint(ranges: pd.DataFrame, key=("customer_id", "symbol")) -> pd.Series:
    """SHA-256 (first 16 hex characters) of each unit's ranges, in date order."""
    key = list(key)
    out = {}
    for k, g in ranges.sort_values("start").groupby(key, sort=True):
        text = ";".join(f"{s:%Y-%m-%d}|{e:%Y-%m-%d}|{float(q):.4f}"
                        for s, e, q in zip(g["start"], g["end"], g["quantity"]))
        out[k] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    s = pd.Series(out, name="fingerprint", dtype=object)
    if len(key) > 1 and len(s):
        s.index = pd.MultiIndex.from_tuples(s.index, names=key)
    return s


def dataset_fingerprint(fingerprints: pd.Series) -> str:
    """One hash for a whole table: the unit fingerprints in key order, hashed again."""
    text = ";".join(f"{k}={v}" for k, v in sorted(fingerprints.items(), key=lambda kv: str(kv[0])))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compare_fingerprints(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Units whose fingerprints differ or exist on one side only."""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1)
    df["status"] = "MATCH"
    df.loc[df["a"].isna(), "status"] = "ONLY_IN_B"
    df.loc[df["b"].isna(), "status"] = "ONLY_IN_A"
    df.loc[df["a"].notna() & df["b"].notna() & (df["a"] != df["b"]), "status"] = "DIFFERENT"
    return df[df["status"] != "MATCH"].reset_index()


def anomalies(ranges: pd.DataFrame, flicker_days: int = 1) -> pd.DataFrame:
    """Ranges that look wrong:

    * ``NEGATIVE_HOLDING``: a share count below zero, impossible for a cash account;
    * ``FLICKER``: a range of at most ``flicker_days`` between two ranges with the same
      quantity (a one-day leave-and-return, or a value that briefly dropped and came back).
    """
    rows = []
    for k, g in ranges.sort_values("start").groupby(["customer_id", "symbol"], sort=True):
        g = g.reset_index(drop=True)
        for i, r in g.iterrows():
            if r["quantity"] < 0:
                rows.append(dict(r, anomaly="NEGATIVE_HOLDING"))
            if 0 < i < len(g) - 1 and r["days"] <= flicker_days and \
                    g.loc[i - 1, "quantity"] == g.loc[i + 1, "quantity"]:
                rows.append(dict(r, anomaly="FLICKER"))
    return pd.DataFrame(rows, columns=RANGE_COLUMNS + ["anomaly"])


def compression_ratio(daily_rows: int, ranges: pd.DataFrame) -> float:
    """How many daily rows each stored range replaces."""
    return round(daily_rows / max(len(ranges), 1), 2)
