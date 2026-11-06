"""Family D: portfolio characteristics. Descriptive only, never reported before Family A."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import model as m
from ..pipeline import portfolio_values
from .base import CheckResult, check
from .common import dedupe_daily, live_events


@check(id="D1", title="Population definition", family="D", scope="profile", severity="info",
       owner="Product (portfolio)",
       question="Who counts as a customer with holdings? (the denominator for everything else)")
def d1_population(ds) -> CheckResult:
    spec = d1_population.spec
    everyone = set(ds["customers"]["customer_id"])
    rows = []
    for table in ("positions_latest", "depository_latest"):
        df = ds[table]
        holding = set(df.loc[df["quantity"] != 0, "customer_id"])
        with_rows = set(df["customer_id"])
        rows.append(dict(route=table, holding=len(holding), rows_nothing_held=len(with_rows - holding),
                         no_rows=len(everyone - with_rows)))
    return CheckResult(spec, pd.DataFrame(), denominator=len(everyone),
                       observations=pd.DataFrame(rows),
                       notes=["state which route the denominator comes from; they differ"])


@check(id="D3", title="Portfolio value distribution", family="D", scope="profile",
       severity="info", owner="Product (portfolio)",
       question="How are portfolio values distributed? (percentiles, never the mean)")
def d3_values(ds) -> CheckResult:
    spec = d3_values.spec
    v = portfolio_values(dedupe_daily(ds), ds["prices"], ds.as_of)["value"]
    v = v[v > 0]
    q = [10, 25, 50, 75, 90, 99]
    obs = pd.DataFrame(dict(percentile=q, value=np.percentile(v, q).round() if len(v) else np.nan))
    return CheckResult(spec, pd.DataFrame(), denominator=len(v), observations=obs)


@check(id="D7", title="Rights lifecycle", family="D", scope="profile", severity="info",
       owner="Product (portfolio)",
       question="How many rights were exercised, sold or left to lapse?")
def d7_rights(ds) -> CheckResult:
    spec = d7_rights.spec
    ev = live_events(ds)
    rows = []
    for ca in ds["corporate_actions"].itertuples(index=False):
        if ca.action_type != m.RIGHTS:
            continue
        r = ev[ev["symbol"] == ca.rights_symbol]
        q = lambda t: int(r.loc[r["event_type"] == t, "quantity"].sum())   # noqa: E731
        lapsed = r[r["event_type"] == m.RIGHTS_EXPIRE]
        rows.append(dict(ca_id=ca.ca_id, rights_symbol=ca.rights_symbol,
                         granted=q(m.RIGHTS_INIT), exercised=q(m.RIGHTS_EXERCISE),
                         sold=q(m.SELL), lapsed=q(m.RIGHTS_EXPIRE),
                         customers_lapsed=int(lapsed["customer_id"].nunique())))
    return CheckResult(spec, pd.DataFrame(), denominator=len(rows),
                       observations=pd.DataFrame(rows),
                       notes=["lapsing customers are also an alerting opportunity"])
