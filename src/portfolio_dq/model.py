"""Shared vocabulary: event types, code lists, ISIN helpers and the trading calendar.

Everything here is generic. The names are invented for this package; they describe the
kind of data a retail brokerage keeps, not any particular system.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable

import pandas as pd

# --------------------------------------------------------------------------------------
# Position events (the transactional ledger, one row per event)
# --------------------------------------------------------------------------------------
BUY = "BUY"
SELL = "SELL"
CUSTODIAN_IN = "CUSTODIAN_IN"            # shares moved in from another broker
CUSTODIAN_OUT = "CUSTODIAN_OUT"          # shares moved out to another broker
OPTION_SETTLE_IN = "OPTION_SETTLE_IN"    # physical settlement of an exercised option, shares received
OPTION_SETTLE_OUT = "OPTION_SETTLE_OUT"  # physical settlement, shares delivered
RIGHTS_EXERCISE = "RIGHTS_EXERCISE"      # customer subscribes: rights leave the rights symbol

CASH_DIVIDEND = "CASH_DIVIDEND"          # no share movement, cash amount only
BONUS_GRANT = "BONUS_GRANT"              # entitlement recorded on record date (not yet tradable)
BONUS_ACTIVATE = "BONUS_ACTIVATE"        # bonus shares credited on activation date
RIGHTS_GRANT = "RIGHTS_GRANT"            # marker on the main symbol, carries the entitlement
RIGHTS_INIT = "RIGHTS_INIT"              # rights credited on the rights symbol
RIGHTS_CONVERT = "RIGHTS_CONVERT"        # exercised rights become shares of the main symbol
RIGHTS_EXPIRE = "RIGHTS_EXPIRE"          # unexercised rights lapse at the end of the period

UNEXPLAINED_INCREASE = "UNEXPLAINED_INCREASE"  # gap plug: depository shows more than the ledger
UNEXPLAINED_DECREASE = "UNEXPLAINED_DECREASE"  # gap plug: depository shows less than the ledger

SOURCE_EVENT_TYPES = (BUY, SELL, CUSTODIAN_IN, CUSTODIAN_OUT, OPTION_SETTLE_IN,
                      OPTION_SETTLE_OUT, RIGHTS_EXERCISE)
CA_EVENT_TYPES = (CASH_DIVIDEND, BONUS_GRANT, BONUS_ACTIVATE, RIGHTS_GRANT, RIGHTS_INIT,
                  RIGHTS_CONVERT, RIGHTS_EXPIRE)
GAP_EVENT_TYPES = (UNEXPLAINED_INCREASE, UNEXPLAINED_DECREASE)
EVENT_TYPES = SOURCE_EVENT_TYPES + CA_EVENT_TYPES + GAP_EVENT_TYPES

#: effect of one unit of ``quantity`` on the running balance
DIRECTION = {
    BUY: 1, SELL: -1, CUSTODIAN_IN: 1, CUSTODIAN_OUT: -1, OPTION_SETTLE_IN: 1,
    OPTION_SETTLE_OUT: -1, RIGHTS_EXERCISE: -1,
    CASH_DIVIDEND: 0, BONUS_GRANT: 0, BONUS_ACTIVATE: 1, RIGHTS_GRANT: 0, RIGHTS_INIT: 1,
    RIGHTS_CONVERT: 1, RIGHTS_EXPIRE: -1,
    UNEXPLAINED_INCREASE: 1, UNEXPLAINED_DECREASE: -1,
}
COST_INFLOWS = {BUY, OPTION_SETTLE_IN, CUSTODIAN_IN, UNEXPLAINED_INCREASE, RIGHTS_CONVERT}
FREE_INFLOWS = {BONUS_ACTIVATE, RIGHTS_INIT}
REALISING_OUTFLOWS = {SELL, OPTION_SETTLE_OUT}

ORIGIN_SOURCE = "SOURCE"     # copied from an upstream feed (trades, transfers, settlements)
ORIGIN_DERIVED = "DERIVED"   # produced by the engine (corporate actions, gap plugs)

CALCULATED, PENDING, FAILED = "CALCULATED", "PENDING", "FAILED"

# --------------------------------------------------------------------------------------
# Depository route
# --------------------------------------------------------------------------------------
FEED_SETTLEMENT, FEED_CUSTODIAN, FEED_ADMIN = "SETTLEMENT", "CUSTODIAN_CHANGE", "ADMIN"
ROW_POSITION, ROW_JOIN, ROW_LEAVE, ROW_IGNORE, ROW_CHANGE_MARKET = (
    "POSITION", "JOIN", "LEAVE", "IGNORE", "CHANGE_MARKET")
REG_DEPOSITORY, REG_ADMIN = "DEPOSITORY", "ADMIN"

# --------------------------------------------------------------------------------------
# Corporate actions and reference data
# --------------------------------------------------------------------------------------
DIVIDEND, BONUS, RIGHTS = "DIVIDEND", "BONUS", "RIGHTS"
SUBSCRIPTION_PRICE = 1_000      # IRR paid per new share in a rights issue (par value)
EQUITY, FIXED_INCOME, RIGHT = "EQUITY", "FIXED_INCOME", "RIGHT"

# return engine status codes known to this package; anything else is a finding
RETURN_STATUSES = ("OK", "NO_HOLDINGS", "MISSING_PRICE", "INVALID_START_VALUE")

# cash movement kinds: only the first two are external cash flows for a return calculation
EXTERNAL_FLOW_KINDS = ("DEPOSIT", "WITHDRAWAL")
INTERNAL_FLOW_KINDS = ("DIVIDEND_CREDIT", "RIGHTS_PAYMENT", "CREDIT_DRAWDOWN", "CREDIT_REPAYMENT")

# --------------------------------------------------------------------------------------
# Check scopes: the labels that keep within-chain and cross-chain findings apart
# --------------------------------------------------------------------------------------
SCOPE_LABELS = {
    "gate": "VALIDITY GATE",
    "within_chain1": "WITHIN-CHAIN (depository route)",
    "within_chain2": "WITHIN-CHAIN (transactional route)",
    "cross_chain": "CROSS-CHAIN",
    "reference": "REFERENCE DATA",
    "cross_system": "CROSS-SYSTEM (return engine / app)",
    "profile": "PROFILE",
}
SCOPE_ORDER = list(SCOPE_LABELS)
SCOPE_GUIDANCE = {
    "gate": "Run first. If these fail, joins in later checks are unsound.",
    "within_chain1": "Derivation defects inside the depository route. Unambiguous; assign to engineering.",
    "within_chain2": "Derivation defects inside the transactional route. Unambiguous; assign to engineering.",
    "cross_chain": "Reconciliation findings. May be a bug, a timing lag or a definitional difference; "
                   "interpret before calling them defects.",
    "reference": "Market and reference data used by the checks above and by the app.",
    "cross_system": "Does the customer see the right number? Downstream consumers of the holdings.",
    "profile": "Portfolio characteristics. Descriptive only, and provisional while Family A fails.",
}


def signed_quantity(event_type: str, quantity: float) -> float:
    """Effect of an event on the running balance (``quantity`` is unsigned)."""
    return DIRECTION[event_type] * quantity


def entitlement(quantity: float, ratio: float) -> int:
    """Shares or rights granted for ``quantity`` held at a ``ratio`` (floored, never negative)."""
    if quantity is None or quantity <= 0:
        return 0
    return int(math.floor(quantity * ratio + 1e-9))


def adjusted_base_price(prev_close: float, actions: Iterable[tuple[str, float]]) -> float:
    """Previous close adjusted for corporate actions taking effect today.

    The daily price limit applies to the adjusted base, so an ex-dividend or ex-bonus drop
    is not a limit break. ``actions`` holds ``(action_type, value)`` pairs.
    """
    base = float(prev_close)
    for action_type, value in actions:
        if action_type == DIVIDEND:
            base -= value
        elif action_type == BONUS:
            base /= 1 + value
        elif action_type == RIGHTS:
            base = (base + SUBSCRIPTION_PRICE * value) / (1 + value)
    return base


# --------------------------------------------------------------------------------------
# ISIN helpers. Canonical security key = first 11 characters (the 12th is a check digit).
# --------------------------------------------------------------------------------------
def isin_check_digit(body: str) -> str:
    """ISO 6166 check digit for the first 11 characters of an ISIN (Luhn on letter values)."""
    digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in body.upper())
    total, double = 0, True
    for ch in reversed(digits):
        x = int(ch) * (2 if double else 1)
        total += x // 10 + x % 10
        double = not double
    return str((10 - total % 10) % 10)


def make_isin(body: str) -> str:
    """Append the correct check digit to an 11-character ISIN body."""
    if len(body) != 11:
        raise ValueError("an ISIN body has 11 characters")
    return body + isin_check_digit(body)


def canonical_isin(isin) -> str | None:
    """Canonical security key: the first 11 characters. Never join on the full 12."""
    if not isinstance(isin, str) or not isin:
        return None
    return isin[:11]


def isin_issues(isin) -> list[str]:
    """Structural problems: LENGTH, PREFIX (serious) and CHECK_DIGIT (known variant issue)."""
    if not isinstance(isin, str):
        return ["LENGTH"]
    issues = []
    if len(isin) != 12:
        issues.append("LENGTH")
    if not isin.upper().startswith("IR"):
        issues.append("PREFIX")
    if len(isin) == 12 and isin[:11].isalnum() and isin[11] != isin_check_digit(isin[:11]):
        issues.append("CHECK_DIGIT")
    return issues


# --------------------------------------------------------------------------------------
# Trading calendar: the Tehran market trades Saturday to Wednesday
# --------------------------------------------------------------------------------------
WEEKMASK = "Sat Sun Mon Tue Wed"


def business_days(start, periods: int, holidays: Iterable = ()) -> pd.DatetimeIndex:
    """``periods`` trading days from ``start`` (Thursday and Friday are the weekend)."""
    return pd.bdate_range(start, periods=periods, freq="C", weekmask=WEEKMASK,
                          holidays=list(holidays))


class Calendar:
    """Ordered trading days with shifting and counting by business day."""

    def __init__(self, days: Iterable):
        self.days = pd.DatetimeIndex(sorted(pd.to_datetime(list(days)))).unique()
        self._values = list(self.days)
        self._pos = {d: i for i, d in enumerate(self._values)}

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    @property
    def as_of(self) -> pd.Timestamp:
        return self._values[-1]

    def index(self, day) -> int:
        """Position of the last trading day on or before ``day`` (-1 if before the start)."""
        day = pd.Timestamp(day)
        if day in self._pos:
            return self._pos[day]
        return bisect.bisect_right(self._values, day) - 1

    def shift(self, day, k: int) -> pd.Timestamp | None:
        """Trading day ``k`` business days after (k > 0) or before (k < 0) ``day``."""
        i = self.index(day) + k
        if 0 <= i < len(self._values):
            return self._values[i]
        return None

    def between(self, a, b) -> int:
        """Business days from ``a`` to ``b`` (negative when ``b`` is earlier)."""
        return self.index(b) - self.index(a)
