from __future__ import annotations

import numpy as np
import pandas as pd

from ou_skew.models.ou import fit_ou_or_stationary_proxy_from_frame


def rolling_ou_zscores(
    skew: pd.DataFrame,
    *,
    ticker: str,
    window: int = 126,
    min_obs: int = 40,
    trading_days_per_year: int = 252,
    forecast_horizon_days: int = 30,
) -> pd.DataFrame:
    """Compute rolling OU z-scores for one ticker's skew series.

    For each date t, the OU model is fit on the previous `window` observations,
    then the current skew is scored against that fitted stationary distribution.
    """
    required = {"date", "ticker", "skew"}
    missing = required.difference(skew.columns)
    if missing:
        raise ValueError(f"Skew frame is missing required columns: {sorted(missing)}")

    df = skew[skew["ticker"].str.upper() == ticker.upper()].copy()
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < max(window, min_obs) + 1:
        raise ValueError(
            f"Not enough observations for {ticker.upper()}: need at least "
            f"{max(window, min_obs) + 1}, got {len(df)}."
        )

    rows: list[dict] = []
    for i in range(window, len(df)):
        hist = df.iloc[i - window : i]
        current = df.iloc[i]
        current_skew = float(current["skew"])

        try:
            fit = fit_ou_or_stationary_proxy_from_frame(
                hist,
                trading_days_per_year=trading_days_per_year,
                min_obs=min_obs,
            )
            expected = fit.expected_value(
                current_skew,
                horizon_days=forecast_horizon_days,
                trading_days_per_year=trading_days_per_year,
            )
            row = {
                "date": current["date"],
                "ticker": ticker.upper(),
                "skew": current_skew,
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
                "n_obs": fit.n_obs,
                "r2": fit.r2,
                "status": fit.method,
                "error": fit.warning,
            }
        except Exception as exc:  # noqa: BLE001 - keep the rolling series intact
            row = {
                "date": current["date"],
                "ticker": ticker.upper(),
                "skew": current_skew,
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
                "n_obs": len(hist),
                "r2": np.nan,
                "status": "failed",
                "error": str(exc),
            }
        rows.append(row)

    return pd.DataFrame(rows)
