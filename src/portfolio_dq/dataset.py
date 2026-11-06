"""The synthetic schema and a small container that loads it from SQLite or CSV.

Two routes produce holdings for each customer and symbol:

* depository route: ``depository_feed`` -> ``depository_daily`` -> ``depository_latest``
* transactional route: source events + corporate actions -> ``position_events`` (one row per
  event) -> ``positions_daily`` (one row per customer, symbol and day) -> ``positions_latest``

Downstream consumers: ``return_outputs`` and ``return_cash_flows`` (return engine) and
``app_display`` (what the customer sees).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .model import Calendar, canonical_isin

POSITION_COLUMNS = [
    ("customer_id", "TEXT"), ("symbol", "TEXT"), ("date", "TEXT"), ("quantity", "INTEGER"),
    ("bought_total", "INTEGER"), ("sold_total", "INTEGER"), ("bonus_total", "INTEGER"),
    ("rights_total", "INTEGER"), ("transfer_in_total", "INTEGER"), ("transfer_out_total", "INTEGER"),
    ("avg_cost", "REAL"), ("realized_pnl", "REAL"), ("calc_version", "INTEGER"),
]

SCHEMA: dict[str, list[tuple[str, str]]] = {
    "calendar": [("date", "TEXT")],
    "customers": [("customer_id", "TEXT"), ("opened_on", "TEXT")],
    "symbols": [("symbol", "TEXT"), ("isin", "TEXT"), ("name", "TEXT"),
                ("instrument_class", "TEXT"), ("parent_symbol", "TEXT")],
    "isin_lineage": [("old_isin", "TEXT"), ("new_isin", "TEXT"), ("effective_date", "TEXT")],
    "fee_schedule": [("instrument_class", "TEXT"), ("sell_fee", "REAL"), ("daily_limit", "REAL")],
    "prices": [("symbol", "TEXT"), ("date", "TEXT"), ("close", "REAL")],
    "corporate_actions": [("ca_id", "TEXT"), ("symbol", "TEXT"), ("action_type", "TEXT"),
                          ("session_date", "TEXT"), ("record_date", "TEXT"),
                          ("activation_date", "TEXT"), ("period_end", "TEXT"), ("value", "REAL"),
                          ("rights_symbol", "TEXT")],
    "depository_feed": [("feed_id", "TEXT"), ("customer_id", "TEXT"), ("isin", "TEXT"),
                        ("date", "TEXT"), ("quantity", "INTEGER"), ("origin", "TEXT")],
    "depository_daily": [("customer_id", "TEXT"), ("symbol", "TEXT"), ("date", "TEXT"),
                         ("quantity", "INTEGER"), ("received_isin", "TEXT"), ("row_type", "TEXT"),
                         ("registration_type", "TEXT")],
    "depository_latest": [("customer_id", "TEXT"), ("symbol", "TEXT"), ("date", "TEXT"),
                          ("quantity", "INTEGER"), ("row_type", "TEXT")],
    "position_events": [("event_id", "INTEGER"), ("customer_id", "TEXT"), ("symbol", "TEXT"),
                        ("date", "TEXT"), ("time", "TEXT"), ("event_type", "TEXT"),
                        ("origin", "TEXT"), ("source_ref", "TEXT"), ("quantity", "INTEGER"),
                        ("price", "REAL"), ("balance_after", "INTEGER"), ("avg_cost_after", "REAL"),
                        ("ca_id", "TEXT"), ("ca_value", "REAL"), ("cash_amount", "REAL"),
                        ("is_deleted", "INTEGER"), ("calc_status", "TEXT")],
    "positions_daily": POSITION_COLUMNS,
    "positions_latest": POSITION_COLUMNS,
    "return_outputs": [("customer_id", "TEXT"), ("as_of", "TEXT"), ("horizon", "TEXT"),
                       ("status", "TEXT"), ("start_value", "REAL"), ("end_value", "REAL")],
    "cash_movements": [("movement_id", "TEXT"), ("customer_id", "TEXT"), ("date", "TEXT"),
                       ("amount", "REAL"), ("kind", "TEXT")],
    "return_cash_flows": [("customer_id", "TEXT"), ("date", "TEXT"), ("amount", "REAL")],
    "app_display": [("customer_id", "TEXT"), ("symbol", "TEXT"), ("as_of", "TEXT"),
                    ("quantity", "INTEGER"), ("close", "REAL"), ("market_value", "REAL"),
                    ("cost_value", "REAL"), ("unrealized_pnl", "REAL"), ("return_pct", "REAL"),
                    ("break_even_price", "REAL")],
}
MANIFEST_TABLE = "injected_defects"
MANIFEST_COLUMNS = ["name", "stage", "target", "description", "expected_checks", "benign"]
DATE_COLUMNS = {"date", "opened_on", "effective_date", "session_date", "record_date",
                "activation_date", "period_end", "as_of"}


def ddl(table: str) -> str:
    cols = ", ".join(f"{c} {t}" for c, t in SCHEMA[table])
    return f"CREATE TABLE {table} ({cols})"


def _empty(table: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c, _ in SCHEMA[table]})


def _to_storage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in DATE_COLUMNS:
            out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
    return out


def _from_storage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in DATE_COLUMNS:
            out[c] = pd.to_datetime(out[c])
    return out


class Dataset:
    """A set of named tables (pandas DataFrames) plus the manifest of injected defects.

    Checks read tables with ``ds["table"]`` or run SQL with ``ds.sql(...)`` against an
    in-memory SQLite copy. The manifest is for tests and reports only; checks never read it.
    """

    def __init__(self, tables: dict[str, pd.DataFrame], manifest: list[dict] | None = None):
        self.tables = {name: tables.get(name, _empty(name)) for name in SCHEMA}
        self.manifest = list(manifest or [])
        self._conn: sqlite3.Connection | None = None

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    @property
    def calendar(self) -> Calendar:
        return Calendar(self.tables["calendar"]["date"])

    @property
    def as_of(self) -> pd.Timestamp:
        return self.calendar.as_of

    def canonical_map(self) -> dict[str, str]:
        """symbol -> canonical ISIN (first 11 characters); unknown symbols map to themselves."""
        s = self.tables["symbols"]
        return {sym: canonical_isin(isin) or sym for sym, isin in zip(s["symbol"], s["isin"])}

    # ---- SQL access --------------------------------------------------------------------
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(":memory:")
            write_tables(self._conn, self.tables, self.manifest)
        return self._conn

    def sql(self, query: str, params=()) -> pd.DataFrame:
        """Run a query against an in-memory SQLite copy; date columns come back as Timestamps."""
        return _from_storage(pd.read_sql_query(query, self.connection(), params=params))

    # ---- persistence -------------------------------------------------------------------
    def to_sqlite(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        with sqlite3.connect(path) as conn:
            write_tables(conn, self.tables, self.manifest)
        return path

    def to_csv(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, df in self.tables.items():
            _to_storage(df).to_csv(directory / f"{name}.csv", index=False)
        pd.DataFrame(self.manifest, columns=MANIFEST_COLUMNS).to_csv(
            directory / f"{MANIFEST_TABLE}.csv", index=False)
        return directory

    @classmethod
    def load(cls, path) -> "Dataset":
        """Load from a SQLite file (opened read-only) or a directory of CSV files."""
        path = Path(path)
        if path.is_dir():
            tables = {}
            for name in SCHEMA:
                f = path / f"{name}.csv"
                tables[name] = _from_storage(pd.read_csv(f)) if f.exists() else _empty(name)
            m = path / f"{MANIFEST_TABLE}.csv"
            manifest = pd.read_csv(m).fillna("").to_dict("records") if m.exists() else []
            return cls(tables, manifest)
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            return cls.from_connection(conn)
        finally:
            conn.close()

    @classmethod
    def from_connection(cls, conn: sqlite3.Connection) -> "Dataset":
        """Read every schema table (and the manifest, if present) from an open connection."""
        tables = {name: _from_storage(pd.read_sql_query(f"SELECT * FROM {name}", conn))
                  for name in SCHEMA}
        try:
            manifest = pd.read_sql_query(f"SELECT * FROM {MANIFEST_TABLE}", conn).fillna("")
            manifest = manifest.to_dict("records")
        except Exception:
            manifest = []
        return cls(tables, manifest)

    def with_sql(self, statements: str) -> "Dataset":
        """A modified copy: run SQL statements (e.g. a test tamper) on a fresh in-memory copy.
        The original dataset is not changed."""
        conn = sqlite3.connect(":memory:")
        try:
            write_tables(conn, self.tables, self.manifest)
            conn.executescript(statements)
            return Dataset.from_connection(conn)
        finally:
            conn.close()


def write_tables(conn: sqlite3.Connection, tables: dict[str, pd.DataFrame],
                 manifest: list[dict] | None = None) -> None:
    for name in SCHEMA:
        conn.execute(ddl(name))
        df = tables.get(name)
        if df is not None and len(df):
            cols = [c for c, _ in SCHEMA[name]]
            _to_storage(df[cols]).to_sql(name, conn, if_exists="append", index=False)
    rows = pd.DataFrame(manifest or [], columns=MANIFEST_COLUMNS).astype(str)
    rows.to_sql(MANIFEST_TABLE, conn, if_exists="replace", index=False)
    conn.commit()
