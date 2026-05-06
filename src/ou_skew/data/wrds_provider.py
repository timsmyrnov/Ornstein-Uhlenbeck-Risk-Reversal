from __future__ import annotations

from dataclasses import dataclass
from getpass import getuser
from pathlib import Path
import re
from typing import Callable, Iterable

import pandas as pd

from ou_skew.data.schema import normalize_option_chain


class WRDSOptionMetricsError(RuntimeError):
    pass


_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-_]+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class WRDSTableConfig:
    """WRDS OptionMetrics table configuration.

    WRDS exposes IvyDB option and security-price tables as year-partitioned
    tables on many accounts, e.g. optionm.opprcd2025 and optionm.secprd2025.
    The default "auto" values resolve to those year-specific names from the
    requested start/end dates.
    """

    library: str = "optionm"
    option_table: str = "auto"
    name_table: str = "secnmd"
    price_table: str = "auto"

    @property
    def option_ref(self) -> str:
        return f"{self.library}.{self.option_table}"

    @property
    def name_ref(self) -> str:
        return f"{self.library}.{self.name_table}"

    @property
    def price_ref(self) -> str:
        return f"{self.library}.{self.price_table}"

    def resolved_for_dates(self, *, start: str, end: str) -> "WRDSTableConfig":
        """Resolve auto table names for a single calendar year request."""
        start = _validate_date_string(start, "start")
        end = _validate_date_string(end, "end")
        start_year = start[:4]
        end_year = end[:4]
        if start_year != end_year and (self.option_table == "auto" or self.price_table == "auto"):
            raise WRDSOptionMetricsError(
                "Auto WRDS table selection currently requires start and end in the same calendar year. "
                "Run one year at a time or explicitly pass --option-table and --price-table."
            )
        option_table = f"opprcd{start_year}" if self.option_table == "auto" else self.option_table
        price_table = f"secprd{start_year}" if self.price_table == "auto" else self.price_table
        return WRDSTableConfig(
            library=self.library,
            option_table=option_table,
            name_table=self.name_table,
            price_table=price_table,
        )


def _validate_date_string(value: str, name: str) -> str:
    if not _DATE_RE.match(value):
        raise ValueError(f"{name} must be YYYY-MM-DD, got {value!r}")
    return value



def _iter_date_chunks(start: str, end: str, chunk_days: int) -> list[tuple[str, str]]:
    if chunk_days <= 0:
        return [(start, end)]
    start_ts = pd.Timestamp(_validate_date_string(start, "start"))
    end_ts = pd.Timestamp(_validate_date_string(end, "end"))
    if end_ts < start_ts:
        raise ValueError(f"end must be on or after start; got {start!r} to {end!r}")
    chunks: list[tuple[str, str]] = []
    cur = start_ts
    while cur <= end_ts:
        chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), end_ts)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks

def _format_ticker_list(tickers: Iterable[str]) -> str:
    cleaned: list[str] = []
    for ticker in tickers:
        t = str(ticker).upper().strip()
        if not t:
            continue
        if not _TICKER_RE.match(t):
            raise ValueError(f"Invalid ticker for WRDS SQL query: {ticker!r}")
        cleaned.append(t)
    if not cleaned:
        raise ValueError("Provide at least one ticker.")
    return ", ".join("'" + t.replace("'", "''") + "'" for t in sorted(set(cleaned)))


def _connect_wrds(username: str | None = None):
    try:
        import wrds  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise WRDSOptionMetricsError(
            "The 'wrds' package is not installed. Run: python3 -m pip install -e '.[dev,wrds]'"
        ) from exc
    return wrds.Connection(wrds_username=username) if username else wrds.Connection()


def default_wrds_username() -> str:
    try:
        return getuser()
    except Exception:
        return ""


class WRDSOptionMetricsProvider:
    """Load historical OptionMetrics/IvyDB option chains from WRDS."""

    def __init__(self, *, username: str | None = None, connection=None, tables: WRDSTableConfig | None = None) -> None:
        self.username = username
        self._connection = connection
        self.tables = tables or WRDSTableConfig()

    @property
    def connection(self):
        if self._connection is None:
            self._connection = _connect_wrds(self.username)
        return self._connection

    def resolved_tables(self, *, start: str, end: str) -> WRDSTableConfig:
        return self.tables.resolved_for_dates(start=start, end=end)

    def build_query(self, *, tickers: Iterable[str], start: str, end: str) -> str:
        start = _validate_date_string(start, "start")
        end = _validate_date_string(end, "end")
        tables = self.resolved_tables(start=start, end=end)
        ticker_sql = _format_ticker_list(tickers)
        return f"""
WITH ids AS (
    SELECT DISTINCT secid, UPPER(ticker) AS ticker
    FROM {tables.name_ref}
    WHERE UPPER(ticker) IN ({ticker_sql})
)
SELECT
    o.date AS date,
    ids.ticker AS ticker,
    o.exdate AS expiry,
    o.cp_flag AS option_type,
    o.strike_price / 1000.0 AS strike,
    s.close AS underlying_price,
    o.impl_volatility AS implied_volatility,
    o.delta AS delta,
    o.best_bid AS bid,
    o.best_offer AS ask,
    o.volume AS volume,
    o.open_interest AS open_interest
FROM {tables.option_ref} AS o
JOIN ids ON o.secid = ids.secid
LEFT JOIN {tables.price_ref} AS s
    ON o.secid = s.secid
   AND o.date = s.date
WHERE o.date BETWEEN '{start}' AND '{end}'
  AND o.exdate > o.date
  AND o.cp_flag IN ('C', 'P')
  AND o.impl_volatility IS NOT NULL
  AND o.delta IS NOT NULL
  AND o.best_bid IS NOT NULL
  AND o.best_offer IS NOT NULL
ORDER BY ids.ticker, o.date, o.exdate, o.cp_flag, o.strike_price
""".strip()

    def raw_load(self, *, tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
        query = self.build_query(tickers=tickers, start=start, end=end)
        try:
            return self.connection.raw_sql(query, date_cols=["date", "expiry"])
        except Exception as exc:  # pragma: no cover
            tables = self.resolved_tables(start=start, end=end)
            raise WRDSOptionMetricsError(
                "WRDS OptionMetrics query failed. Common causes: missing OptionMetrics access, "
                "different table names, or unavailable columns. The query used "
                f"{tables.option_ref}, {tables.name_ref}, and {tables.price_ref}. "
                "Override with --library/--option-table/--name-table/--price-table if needed.\n\n"
                f"Original error: {exc}"
            ) from exc

    def load(self, *, tickers: Iterable[str], start: str, end: str, normalize_iv_scale: bool = True) -> pd.DataFrame:
        raw = self.raw_load(tickers=tickers, start=start, end=end)
        if raw.empty:
            raise WRDSOptionMetricsError(
                f"WRDS returned zero rows for {list(tickers)} from {start} to {end}."
            )
        return normalize_option_chain(raw, tickers=tickers, start=start, end=end, normalize_iv_scale=normalize_iv_scale)

    def load_chunked(
        self,
        *,
        tickers: Iterable[str],
        start: str,
        end: str,
        chunk_days: int = 31,
        normalize_iv_scale: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        ticker_list = list(tickers)
        chunks = _iter_date_chunks(start, end, chunk_days)
        frames: list[pd.DataFrame] = []
        for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            tables = self.resolved_tables(start=chunk_start, end=chunk_end)
            if progress is not None:
                progress(
                    f"[{i}/{len(chunks)}] WRDS query {chunk_start} to {chunk_end} "
                    f"using {tables.option_ref} + {tables.price_ref} ..."
                )
            raw = self.raw_load(tickers=ticker_list, start=chunk_start, end=chunk_end)
            if progress is not None:
                progress(f"    got {len(raw):,} option rows")
            if not raw.empty:
                frames.append(raw)
        if not frames:
            raise WRDSOptionMetricsError(
                f"WRDS returned zero rows for {ticker_list} from {start} to {end}."
            )
        raw_all = pd.concat(frames, ignore_index=True)
        return normalize_option_chain(raw_all, tickers=ticker_list, start=start, end=end, normalize_iv_scale=normalize_iv_scale)

    def fetch_to_csv(
        self,
        *,
        tickers: Iterable[str],
        start: str,
        end: str,
        out: str | Path,
        chunk_days: int = 31,
        progress: Callable[[str], None] | None = None,
    ) -> Path:
        if chunk_days > 0:
            df = self.load_chunked(tickers=tickers, start=start, end=end, chunk_days=chunk_days, progress=progress)
        else:
            df = self.load(tickers=tickers, start=start, end=end)
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path
