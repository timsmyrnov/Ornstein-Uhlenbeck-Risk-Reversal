from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ou_skew.models.ou import OUFit, fit_ou_or_stationary_proxy_from_frame, simulate_ou


def _ticker_frame(skew: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = skew[skew["ticker"].str.upper() == ticker.upper()].copy()
    frame = frame.sort_values("date")
    if frame.empty:
        raise ValueError(f"No skew data found for ticker {ticker}")
    return frame


def plot_skew_with_ou_mean(
    skew: pd.DataFrame,
    *,
    ticker: str,
    out_path: str | Path,
    min_obs: int = 20,
) -> Path:
    frame = _ticker_frame(skew, ticker)
    fit = fit_ou_or_stationary_proxy_from_frame(frame, min_obs=min_obs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame["date"], frame["skew"], label="skew")
    theta_label = "OU theta" if fit.method == "ou_ar1" else "proxy mean"
    ax.axhline(fit.theta, linestyle="--", label=theta_label)
    title_suffix = "OU mean" if fit.method == "ou_ar1" else "stationary proxy mean"
    ax.set_title(f"{ticker.upper()} implied-vol skew with {title_suffix}")
    if fit.warning:
        ax.text(0.01, 0.01, fit.method, transform=ax.transAxes, fontsize=8, va="bottom")
    ax.set_xlabel("Date")
    ax.set_ylabel("Skew")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def simulate_ou_paths_from_skew(
    skew: pd.DataFrame,
    *,
    ticker: str,
    n_steps: int = 63,
    n_paths: int = 100,
    lookback_days: int | None = None,
    min_obs: int = 20,
    trading_days_per_year: int = 252,
    seed: int | None = 7,
) -> tuple[pd.DataFrame, OUFit]:
    """Fit OU to a ticker's skew series and simulate future paths.

    Returns a long DataFrame with columns date, ticker, path, step, skew and the fitted OU parameters.
    """
    frame = _ticker_frame(skew, ticker)
    if lookback_days:
        latest_date = pd.Timestamp(frame["date"].max())
        cutoff = latest_date - pd.Timedelta(days=lookback_days)
        fit_frame = frame[frame["date"] >= cutoff]
    else:
        fit_frame = frame

    fit = fit_ou_or_stationary_proxy_from_frame(
        fit_frame,
        trading_days_per_year=trading_days_per_year,
        min_obs=min_obs,
    )

    x0 = float(frame["skew"].iloc[-1])
    last_date = pd.Timestamp(frame["date"].iloc[-1])
    dates = pd.bdate_range(last_date, periods=n_steps)

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for path_id in range(n_paths):
        path_seed = int(rng.integers(0, 2**32 - 1)) if seed is not None else None
        values = simulate_ou(
            theta=fit.theta,
            kappa=fit.kappa,
            sigma=fit.sigma,
            x0=x0,
            n_steps=n_steps,
            dt_years=fit.dt_years,
            seed=path_seed,
        )
        for step, (date, value) in enumerate(zip(dates, values, strict=True)):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker.upper(),
                    "path": path_id,
                    "step": step,
                    "skew": float(value),
                }
            )

    return pd.DataFrame(rows), fit


def plot_ou_simulated_paths(
    paths: pd.DataFrame,
    *,
    fit: OUFit,
    ticker: str,
    out_path: str | Path,
    max_paths_to_draw: int = 250,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))

    for _, path in paths.groupby("path", sort=False):
        if int(path["path"].iloc[0]) >= max_paths_to_draw:
            continue
        ax.plot(path["date"], path["skew"], alpha=0.15, linewidth=0.8)

    theta_label = "OU theta" if fit.method == "ou_ar1" else "proxy mean"
    ax.axhline(fit.theta, linestyle="--", label=theta_label)
    title = "simulated OU skew paths" if fit.method == "ou_ar1" else "simulated OU-style proxy skew paths"
    ax.set_title(f"{ticker.upper()} {title}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Skew")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_ou_terminal_distribution(
    paths: pd.DataFrame,
    *,
    fit: OUFit,
    ticker: str,
    out_path: str | Path,
    bins: int = 60,
) -> Path:
    terminal = paths.sort_values(["path", "step"]).groupby("path", as_index=False).tail(1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(terminal["skew"], bins=bins)
    theta_label = "OU theta" if fit.method == "ou_ar1" else "proxy mean"
    ax.axvline(fit.theta, linestyle="--", label=theta_label)
    title = "simulated terminal skew distribution" if fit.method == "ou_ar1" else "simulated proxy terminal skew distribution"
    ax.set_title(f"{ticker.upper()} {title}")
    ax.set_xlabel("Terminal skew")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_rolling_zscores(
    zscores: pd.DataFrame,
    *,
    ticker: str,
    out_path: str | Path,
) -> Path:
    frame = zscores[zscores["ticker"].str.upper() == ticker.upper()].copy()
    frame = frame.sort_values("date")
    if frame.empty:
        raise ValueError(f"No z-score data found for ticker {ticker}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame["date"], frame["z_score"], label="OU z-score")
    ax.axhline(2.0, linestyle="--", linewidth=0.8)
    ax.axhline(1.0, linestyle="--", linewidth=0.8)
    ax.axhline(0.0, linestyle="-", linewidth=0.8)
    ax.axhline(-1.0, linestyle="--", linewidth=0.8)
    ax.axhline(-2.0, linestyle="--", linewidth=0.8)
    ax.set_title(f"{ticker.upper()} rolling OU skew z-scores")
    ax.set_xlabel("Date")
    ax.set_ylabel("Z-score")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
