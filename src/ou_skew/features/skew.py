from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

SkewDefinition = Literal["put_minus_call", "call_minus_put", "put_minus_atm"]


@dataclass(frozen=True)
class ContractSelection:
    implied_volatility: float
    delta: float
    strike: float
    expiry: pd.Timestamp
    days_to_expiry: int
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    open_interest: float | None = None


class SkewExtractionError(ValueError):
    pass


def _liquidity_filter(
    df: pd.DataFrame,
    *,
    min_volume: int | None = None,
    min_open_interest: int | None = None,
    max_spread_pct: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if min_volume is not None and "volume" in out.columns:
        out = out[out["volume"].fillna(0) >= min_volume]
    if min_open_interest is not None and "open_interest" in out.columns:
        out = out[out["open_interest"].fillna(0) >= min_open_interest]
    if max_spread_pct is not None and {"bid", "ask"}.issubset(out.columns):
        mid = (out["bid"] + out["ask"]) / 2
        spread_pct = (out["ask"] - out["bid"]) / mid.replace(0, np.nan)
        out = out[(spread_pct <= max_spread_pct) | spread_pct.isna()]
    return out


def select_nearest_contract(
    chain: pd.DataFrame,
    *,
    option_type: str,
    target_delta: float,
    target_tenor_days: int,
    min_volume: int | None = None,
    min_open_interest: int | None = None,
    max_spread_pct: float | None = None,
) -> ContractSelection | None:
    """Select the contract nearest to a target tenor and delta.

    Ranking is lexicographic: first get close to the target tenor, then target delta.
    This keeps the feature anchored to a fixed maturity before optimizing delta.
    """
    option_type = option_type.upper()
    if option_type not in {"C", "P"}:
        raise ValueError("option_type must be 'C' or 'P'")

    sub = chain[chain["option_type"] == option_type].copy()
    sub = _liquidity_filter(
        sub,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        max_spread_pct=max_spread_pct,
    )
    sub = sub.dropna(subset=["implied_volatility", "delta", "days_to_expiry", "strike", "expiry"])
    if sub.empty:
        return None

    sub["tenor_error"] = (sub["days_to_expiry"] - target_tenor_days).abs()
    sub["delta_error"] = (sub["delta"] - target_delta).abs()
    sub = sub.sort_values(["tenor_error", "delta_error", "expiry", "strike"])
    row = sub.iloc[0]

    return ContractSelection(
        implied_volatility=float(row["implied_volatility"]),
        delta=float(row["delta"]),
        strike=float(row["strike"]),
        expiry=pd.Timestamp(row["expiry"]),
        days_to_expiry=int(row["days_to_expiry"]),
        bid=float(row["bid"]) if "bid" in row and pd.notna(row["bid"]) else None,
        ask=float(row["ask"]) if "ask" in row and pd.notna(row["ask"]) else None,
        volume=float(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else None,
        open_interest=float(row["open_interest"])
        if "open_interest" in row and pd.notna(row["open_interest"])
        else None,
    )


def _select_atm_contract(
    chain: pd.DataFrame,
    *,
    target_tenor_days: int,
    min_volume: int | None = None,
    min_open_interest: int | None = None,
    max_spread_pct: float | None = None,
) -> ContractSelection | None:
    sub = _liquidity_filter(
        chain,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        max_spread_pct=max_spread_pct,
    ).copy()
    sub = sub.dropna(
        subset=["implied_volatility", "delta", "days_to_expiry", "strike", "expiry", "underlying_price"]
    )
    if sub.empty:
        return None
    sub["tenor_error"] = (sub["days_to_expiry"] - target_tenor_days).abs()
    sub["moneyness_error"] = (sub["strike"] / sub["underlying_price"] - 1.0).abs()
    sub = sub.sort_values(["tenor_error", "moneyness_error", "expiry", "strike"])
    row = sub.iloc[0]
    return ContractSelection(
        implied_volatility=float(row["implied_volatility"]),
        delta=float(row["delta"]),
        strike=float(row["strike"]),
        expiry=pd.Timestamp(row["expiry"]),
        days_to_expiry=int(row["days_to_expiry"]),
    )


def compute_skew_timeseries(
    options: pd.DataFrame,
    *,
    tickers: Iterable[str] | None = None,
    target_tenor_days: int = 30,
    call_delta: float = 0.25,
    put_delta: float = -0.25,
    skew_definition: SkewDefinition = "put_minus_call",
    min_volume: int | None = None,
    min_open_interest: int | None = None,
    max_spread_pct: float | None = None,
) -> pd.DataFrame:
    """Compute a daily skew feature for each ticker.

    Default definition: 25-delta put IV minus 25-delta call IV.
    """
    if skew_definition not in {"put_minus_call", "call_minus_put", "put_minus_atm"}:
        raise ValueError(f"Unknown skew_definition: {skew_definition}")

    df = options.copy()
    if tickers:
        ticker_set = {t.upper() for t in tickers}
        df = df[df["ticker"].isin(ticker_set)]

    rows: list[dict] = []
    group_cols = ["ticker", "date"]
    for (ticker, date), chain in df.groupby(group_cols, sort=True):
        call = select_nearest_contract(
            chain,
            option_type="C",
            target_delta=call_delta,
            target_tenor_days=target_tenor_days,
            min_volume=min_volume,
            min_open_interest=min_open_interest,
            max_spread_pct=max_spread_pct,
        )
        put = select_nearest_contract(
            chain,
            option_type="P",
            target_delta=put_delta,
            target_tenor_days=target_tenor_days,
            min_volume=min_volume,
            min_open_interest=min_open_interest,
            max_spread_pct=max_spread_pct,
        )
        if call is None or put is None:
            continue

        atm = None
        if skew_definition == "put_minus_atm":
            atm = _select_atm_contract(
                chain,
                target_tenor_days=target_tenor_days,
                min_volume=min_volume,
                min_open_interest=min_open_interest,
                max_spread_pct=max_spread_pct,
            )
            if atm is None:
                continue
            skew = put.implied_volatility - atm.implied_volatility
        elif skew_definition == "put_minus_call":
            skew = put.implied_volatility - call.implied_volatility
        else:
            skew = call.implied_volatility - put.implied_volatility

        rows.append(
            {
                "date": pd.Timestamp(date),
                "ticker": ticker,
                "skew": float(skew),
                "skew_definition": skew_definition,
                "target_tenor_days": target_tenor_days,
                "call_iv": call.implied_volatility,
                "put_iv": put.implied_volatility,
                "atm_iv": atm.implied_volatility if atm else np.nan,
                "call_delta": call.delta,
                "put_delta": put.delta,
                "call_strike": call.strike,
                "put_strike": put.strike,
                "call_expiry": call.expiry,
                "put_expiry": put.expiry,
                "call_days_to_expiry": call.days_to_expiry,
                "put_days_to_expiry": put.days_to_expiry,
            }
        )

    if not rows:
        raise SkewExtractionError("No valid skew observations could be extracted.")

    out = pd.DataFrame(rows)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
