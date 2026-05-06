from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ou_skew.features.skew import compute_skew_timeseries
from ou_skew.models.ou import fit_ou_or_stationary_proxy_from_frame


def scan_ou_skew(
    options: pd.DataFrame,
    *,
    tickers: Iterable[str] | None = None,
    lookback_days: int = 252,
    target_tenor_days: int = 30,
    call_delta: float = 0.25,
    put_delta: float = -0.25,
    skew_definition: str = "put_minus_call",
    min_volume: int | None = None,
    min_open_interest: int | None = None,
    max_spread_pct: float | None = None,
    trading_days_per_year: int = 252,
    forecast_horizon_days: int = 30,
    min_obs: int = 20,
) -> pd.DataFrame:
    skew = compute_skew_timeseries(
        options,
        tickers=tickers,
        target_tenor_days=target_tenor_days,
        call_delta=call_delta,
        put_delta=put_delta,
        skew_definition=skew_definition,  # type: ignore[arg-type]
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        max_spread_pct=max_spread_pct,
    )

    rows: list[dict] = []
    for ticker, frame in skew.groupby("ticker", sort=True):
        frame = frame.sort_values("date")
        if lookback_days:
            latest_date = pd.Timestamp(frame["date"].max())
            cutoff = latest_date - pd.Timedelta(days=lookback_days)
            fit_frame = frame[frame["date"] >= cutoff]
        else:
            fit_frame = frame

        latest = frame.iloc[-1]
        current_skew = float(latest["skew"])
        row = {
            "ticker": ticker,
            "latest_date": latest["date"],
            "current_skew": current_skew,
            "n_obs": int(len(fit_frame)),
            "status": "ok",
            "error": "",
        }
        try:
            fit = fit_ou_or_stationary_proxy_from_frame(
                fit_frame,
                trading_days_per_year=trading_days_per_year,
                min_obs=min_obs,
            )
            expected = fit.expected_value(
                current_skew,
                horizon_days=forecast_horizon_days,
                trading_days_per_year=trading_days_per_year,
            )
            row.update(
                {
                    "theta": fit.theta,
                    "kappa": fit.kappa,
                    "sigma": fit.sigma,
                    "phi": fit.phi,
                    "stationary_std": fit.stationary_std,
                    "half_life_days": fit.half_life_days,
                    "z_score": fit.z_score(current_skew),
                    "expected_skew_horizon": expected,
                    "expected_change_horizon": expected - current_skew,
                    "forecast_horizon_days": forecast_horizon_days,
                    "r2": fit.r2,
                    "status": fit.method,
                    "error": fit.warning,
                }
            )
        except Exception as exc:  # noqa: BLE001 - scanner should continue ticker-by-ticker
            row.update(
                {
                    "theta": np.nan,
                    "kappa": np.nan,
                    "sigma": np.nan,
                    "phi": np.nan,
                    "stationary_std": np.nan,
                    "half_life_days": np.nan,
                    "z_score": np.nan,
                    "expected_skew_horizon": np.nan,
                    "expected_change_horizon": np.nan,
                    "forecast_horizon_days": forecast_horizon_days,
                    "r2": np.nan,
                    "status": "failed",
                    "error": str(exc),
                }
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_z_score"] = out["z_score"].abs()
    return out.sort_values(["status", "abs_z_score"], ascending=[True, False]).reset_index(drop=True)
