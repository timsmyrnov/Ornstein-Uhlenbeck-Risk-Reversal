from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

REQUIRED_OPTION_COLUMNS = {
    "date",
    "ticker",
    "expiry",
    "option_type",
    "strike",
    "underlying_price",
    "implied_volatility",
    "delta",
}

OPTIONAL_OPTION_COLUMNS = {
    "bid",
    "ask",
    "volume",
    "open_interest",
}

COLUMN_ALIASES = {
    # OptionMetrics-style names and common vendor alternatives.
    "secid": "ticker",
    "symbol": "ticker",
    "cp_flag": "option_type",
    "optiontype": "option_type",
    "exdate": "expiry",
    "expiration": "expiry",
    "best_bid": "bid",
    "best_offer": "ask",
    "impl_volatility": "implied_volatility",
    "iv": "implied_volatility",
    "underlying": "underlying_price",
    "spot": "underlying_price",
    "close": "underlying_price",
    "openinterest": "open_interest",
    "oi": "open_interest",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = col.strip().lower()
        renamed[col] = COLUMN_ALIASES.get(key, key)
    return df.rename(columns=renamed)


def validate_option_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_OPTION_COLUMNS - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Option chain data is missing required columns: {missing_cols}")


def normalize_option_chain(
    df: pd.DataFrame,
    *,
    tickers: Iterable[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    normalize_iv_scale: bool = True,
) -> pd.DataFrame:
    """Return a clean option-chain DataFrame with standard column names.

    The normalized schema expects decimal IV, e.g. 0.32 rather than 32.
    If most IVs look like percentages, they are divided by 100.
    """
    out = normalize_columns(df.copy())
    validate_option_schema(out)

    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["expiry"] = pd.to_datetime(out["expiry"]).dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["option_type"] = normalize_option_type(out["option_type"])

    numeric_cols = [
        "strike",
        "underlying_price",
        "implied_volatility",
        "delta",
        "bid",
        "ask",
        "volume",
        "open_interest",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if normalize_iv_scale and out["implied_volatility"].dropna().median() > 3.0:
        out["implied_volatility"] = out["implied_volatility"] / 100.0

    out["days_to_expiry"] = (out["expiry"] - out["date"]).dt.days
    out = out[out["days_to_expiry"] > 0]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=list(REQUIRED_OPTION_COLUMNS) + ["days_to_expiry"])
    out = out[out["implied_volatility"] > 0]

    if tickers:
        ticker_set = {t.upper() for t in tickers}
        out = out[out["ticker"].isin(ticker_set)]

    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start).normalize()]
    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end).normalize()]

    sort_cols = ["ticker", "date", "expiry", "option_type", "strike"]
    return out.sort_values(sort_cols).reset_index(drop=True)


def normalize_option_type(s: pd.Series) -> pd.Series:
    normalized = s.astype(str).str.upper().str.strip()
    normalized = normalized.replace(
        {
            "CALL": "C",
            "C": "C",
            "1": "C",
            "PUT": "P",
            "P": "P",
            "-1": "P",
        }
    )
    invalid = set(normalized.dropna().unique()) - {"C", "P"}
    if invalid:
        raise ValueError(f"Invalid option_type values: {sorted(invalid)}")
    return normalized
