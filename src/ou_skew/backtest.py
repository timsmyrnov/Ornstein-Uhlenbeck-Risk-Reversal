from __future__ import annotations

import numpy as np
import pandas as pd

from ou_skew.models.ou import fit_ou_or_stationary_proxy_from_frame


def rolling_ou_zscore_backtest(
    skew: pd.DataFrame,
    *,
    ticker: str,
    window: int = 126,
    entry_z: float = 1.5,
    exit_z: float = 0.25,
    trading_days_per_year: int = 252,
    min_obs: int = 40,
) -> pd.DataFrame:
    """Simple research backtest on the skew series itself.

    Position convention:
    - z > entry_z means skew is rich versus OU mean, so position = -1.
    - z < -entry_z means skew is cheap versus OU mean, so position = +1.

    PnL is position * next-day change in the skew feature. This is not an
    executable options PnL; it is a signal sanity check.
    """
    df = skew[skew["ticker"].str.upper() == ticker.upper()].copy()
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < max(window, min_obs) + 2:
        raise ValueError("Not enough observations for rolling backtest.")

    rows: list[dict] = []
    position = 0

    for i in range(window, len(df) - 1):
        hist = df.iloc[i - window : i]
        current = df.iloc[i]
        next_row = df.iloc[i + 1]

        try:
            fit = fit_ou_or_stationary_proxy_from_frame(
                hist,
                trading_days_per_year=trading_days_per_year,
                min_obs=min_obs,
            )
            z = fit.z_score(float(current["skew"]))
        except Exception:
            z = np.nan

        if np.isfinite(z):
            if position == 0:
                if z > entry_z:
                    position = -1
                elif z < -entry_z:
                    position = 1
            elif abs(z) < exit_z:
                position = 0

        skew_change = float(next_row["skew"] - current["skew"])
        pnl = position * skew_change
        rows.append(
            {
                "date": current["date"],
                "ticker": ticker.upper(),
                "skew": float(current["skew"]),
                "z_score": z,
                "position": position,
                "next_skew_change": skew_change,
                "signal_pnl": pnl,
                "cum_signal_pnl": np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out["cum_signal_pnl"] = out["signal_pnl"].cumsum()
    return out
