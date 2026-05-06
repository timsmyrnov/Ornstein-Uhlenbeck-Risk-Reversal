from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from typing import Iterable

import numpy as np
import pandas as pd

from ou_skew.data.schema import normalize_option_chain


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def black_scholes_delta(
    *,
    spot: float,
    strike: float,
    tau_years: float,
    volatility: float,
    risk_free_rate: float,
    option_type: str,
) -> float:
    if spot <= 0 or strike <= 0 or tau_years <= 0 or volatility <= 0:
        return np.nan
    d1 = (log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * tau_years) / (
        volatility * sqrt(tau_years)
    )
    if option_type == "C":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


@dataclass(frozen=True)
class YFinanceSnapshotProvider:
    """Download current option-chain snapshots from yfinance.

    This provider is not a historical data source. It is useful if you run it daily
    and append snapshots to your own CSV store.
    """

    risk_free_rate: float = 0.04

    def load(self, *, tickers: Iterable[str]) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError(
                "Install the optional dependency first: pip install -e '.[yfinance]'"
            ) from exc

        rows: list[pd.DataFrame] = []
        snapshot_date = pd.Timestamp.today().normalize()

        for ticker in tickers:
            ticker = ticker.upper().strip()
            yf_ticker = yf.Ticker(ticker)
            history = yf_ticker.history(period="5d")
            if history.empty:
                continue
            spot = float(history["Close"].dropna().iloc[-1])

            for expiry in yf_ticker.options:
                chain = yf_ticker.option_chain(expiry)
                for option_type, frame in [("C", chain.calls), ("P", chain.puts)]:
                    if frame.empty:
                        continue
                    tmp = frame.copy()
                    tmp["date"] = snapshot_date
                    tmp["ticker"] = ticker
                    tmp["expiry"] = pd.Timestamp(expiry).normalize()
                    tmp["option_type"] = option_type
                    tmp["underlying_price"] = spot
                    tmp = tmp.rename(
                        columns={
                            "impliedVolatility": "implied_volatility",
                            "openInterest": "open_interest",
                        }
                    )
                    tau = max((pd.Timestamp(expiry).normalize() - snapshot_date).days, 1) / 365.0
                    tmp["delta"] = [
                        black_scholes_delta(
                            spot=spot,
                            strike=float(k),
                            tau_years=tau,
                            volatility=float(iv),
                            risk_free_rate=self.risk_free_rate,
                            option_type=option_type,
                        )
                        for k, iv in zip(tmp["strike"], tmp["implied_volatility"], strict=False)
                    ]
                    keep = [
                        "date",
                        "ticker",
                        "expiry",
                        "option_type",
                        "strike",
                        "underlying_price",
                        "implied_volatility",
                        "delta",
                        "bid",
                        "ask",
                        "volume",
                        "open_interest",
                    ]
                    rows.append(tmp[[c for c in keep if c in tmp.columns]])

        if not rows:
            return pd.DataFrame()
        return normalize_option_chain(pd.concat(rows, ignore_index=True))

    def save_snapshot(self, *, tickers: Iterable[str], out_path: str) -> pd.DataFrame:
        df = self.load(tickers=tickers)
        if df.empty:
            raise ValueError("No option chains were downloaded.")
        out = pd.DataFrame(df)
        out_path_obj = pd.io.common.stringify_path(out_path)
        out.to_csv(out_path_obj, index=False)
        return out
