"""Masking holdings for a test environment without breaking the ledger arithmetic.

These transformations are NOT anonymization. They retain identifiers and structural
relationships and must never be used to justify publishing real customer records.

Two tools:

* :func:`customer_multiplier`: a deterministic factor between 0.5 and 2.0 per customer,
  from a SHA-256 hash of the customer id. Every quantity of that customer is scaled by the
  same factor, so "sum of movements = balance" still holds - provided running balances are
  recomputed from the masked movements rather than masked separately (rounding each
  number on its own breaks the identity).
* :func:`compress_magnitudes`: a per-symbol power map that pulls the largest and smallest
  holders together (largest / smallest close to a target ratio), because a multiplier
  alone hides the numbers but not who the large holders are. Zeros stay zero and negative
  values keep their sign, so data anomalies remain visible after masking.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import model as m

LOW, HIGH = 500, 2000      # multiplier range in thousandths: 0.500 ... 2.000


def customer_multiplier(customer_id) -> float:
    """Deterministic factor in [0.5, 2.0] for a customer.

    The hash is taken over the *customer* id, never a row id: a per-row factor would give
    each movement a different scale and the movements would no longer add up to the
    balance. Python integers do not overflow, so no sign handling is needed.
    """
    h = int.from_bytes(hashlib.sha256(str(customer_id).encode("utf-8")).digest()[:8], "big")
    return (h % (HIGH - LOW + 1) + LOW) / 1000.0


def mask_ledger(events: pd.DataFrame) -> pd.DataFrame:
    """Scale ledger quantities per customer and recompute running balances.

    ``quantity`` is multiplied and rounded; ``balance_after`` is rebuilt as the running sum
    of the signed masked quantities (soft-deleted and uncalculated rows keep their place but
    do not move the balance), so the continuity check still passes after masking.
    """
    ev = events.sort_values(["customer_id", "symbol", "date", "time", "event_id"],
                            kind="stable").copy()
    k = ev["customer_id"].map(customer_multiplier)
    ev["quantity"] = (ev["quantity"] * k).round().astype("int64")
    if "cash_amount" in ev:
        ev["cash_amount"] = (ev["cash_amount"] * k).round(2)
    counts = ~ev["is_deleted"].astype(bool) & (ev["calc_status"] == m.CALCULATED)
    signed = [m.signed_quantity(t, q) if c else 0
              for t, q, c in zip(ev["event_type"], ev["quantity"], counts)]
    ev["balance_after"] = pd.Series(signed, index=ev.index).groupby(
        [ev["customer_id"], ev["symbol"]]).cumsum()
    ev.loc[~counts, "balance_after"] = events.loc[ev.index[~counts], "balance_after"] * \
        k[~counts]
    return ev.sort_index()


def compress_magnitudes(x: pd.Series, target_ratio: float = 5.0, p_low: float = 5,
                        p_high: float = 95) -> pd.Series:
    """Map positive values so that (p_high value) / (p_low value) is about ``target_ratio``.

    Steps: robust range from percentiles (one outlier cannot set the scale); clip to that
    range; exponent ``p = ln(target) / ln(real ratio)``; affine map of ``x ** p`` onto a
    range centred on the median in log space. Zeros stay zero; negative values are mapped
    by magnitude and keep their sign. Groups already tighter than the target are unchanged.
    """
    x = x.astype(float)
    pos = x[x > 0]
    if len(pos) < 2:
        return x.copy()
    lo, hi = np.percentile(pos, [p_low, p_high])
    lo = max(lo, 1.0)
    if hi <= lo or hi / lo <= target_ratio:
        return x.copy()
    p = np.log(target_ratio) / np.log(hi / lo)
    med = float(np.median(pos))
    y_lo, y_hi = med / np.sqrt(target_ratio), med * np.sqrt(target_ratio)
    b = (y_hi - y_lo) / (hi ** p - lo ** p)
    a = y_lo - b * lo ** p
    mag = x.abs().clip(lo, hi)
    out = np.sign(x) * (a + b * mag ** p)
    out[x == 0] = 0.0
    return out
