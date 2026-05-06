from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def simulate_skew(theta: float, kappa: float, sigma: float, x0: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    phi = np.exp(-kappa * dt)
    innovation_std = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = theta + phi * (x[i - 1] - theta) + innovation_std * rng.normal()
    return x


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-02", periods=320)
    tickers = ["AAPL", "MSFT"]
    rows = []

    for t_i, ticker in enumerate(tickers):
        skew = simulate_skew(theta=0.075 + 0.01 * t_i, kappa=7.0, sigma=0.08, x0=0.03, n=len(dates), seed=10 + t_i)
        spot = 150 + 100 * t_i + np.cumsum(rng.normal(0, 1.2, size=len(dates)))
        for date, s, sk in zip(dates, spot, skew, strict=False):
            base_iv = 0.24 + 0.02 * t_i + 0.015 * rng.normal()
            for tenor in [25, 32, 60]:
                expiry = date + pd.Timedelta(days=tenor)
                tenor_adj = 0.0004 * (tenor - 30)
                contracts = [
                    ("C", 0.10, s * 1.12, base_iv - 0.65 * sk + tenor_adj),
                    ("C", 0.25, s * 1.06, base_iv - 0.50 * sk + tenor_adj),
                    ("C", 0.50, s * 1.00, base_iv + tenor_adj),
                    ("P", -0.50, s * 1.00, base_iv + tenor_adj),
                    ("P", -0.25, s * 0.94, base_iv + 0.50 * sk + tenor_adj),
                    ("P", -0.10, s * 0.88, base_iv + 0.70 * sk + tenor_adj),
                ]
                for option_type, delta, strike, iv in contracts:
                    mid = max(0.25, iv * s * 0.04)
                    spread = 0.04 + 0.02 * rng.random()
                    rows.append(
                        {
                            "date": date.date(),
                            "ticker": ticker,
                            "expiry": expiry.date(),
                            "option_type": option_type,
                            "strike": round(float(strike), 2),
                            "underlying_price": round(float(s), 2),
                            "implied_volatility": round(float(max(iv, 0.05)), 6),
                            "delta": delta,
                            "bid": round(mid - spread / 2, 2),
                            "ask": round(mid + spread / 2, 2),
                            "volume": int(rng.integers(1, 500)),
                            "open_interest": int(rng.integers(10, 5000)),
                        }
                    )

    out = pd.DataFrame(rows)
    path = Path(__file__).resolve().parents[1] / "data" / "sample_option_chains.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {len(out):,} rows to {path}")


if __name__ == "__main__":
    main()
