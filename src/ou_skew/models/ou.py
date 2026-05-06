from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OUFit:
    theta: float
    kappa: float
    sigma: float
    phi: float
    intercept: float
    residual_std: float
    stationary_std: float
    half_life_days: float
    dt_years: float
    n_obs: int
    r2: float
    method: str = "ou_ar1"
    warning: str = ""

    def z_score(self, x: float) -> float:
        if self.stationary_std <= 0 or not np.isfinite(self.stationary_std):
            return np.nan
        return (x - self.theta) / self.stationary_std

    def expected_value(self, x0: float, *, horizon_days: int, trading_days_per_year: int = 252) -> float:
        horizon_years = horizon_days / trading_days_per_year
        return self.theta + np.exp(-self.kappa * horizon_years) * (x0 - self.theta)

    def expected_change(self, x0: float, *, horizon_days: int, trading_days_per_year: int = 252) -> float:
        return self.expected_value(
            x0, horizon_days=horizon_days, trading_days_per_year=trading_days_per_year
        ) - x0


def _prepare_values(values: pd.Series | np.ndarray | list[float], *, min_obs: int) -> np.ndarray:
    x = pd.Series(values, dtype="float64").dropna().to_numpy()
    if len(x) < min_obs:
        raise ValueError(f"Need at least {min_obs} observations to fit OU; got {len(x)}")
    return x


def _ar1_ols(x: np.ndarray) -> tuple[float, float, np.ndarray, float, float]:
    x_prev = x[:-1]
    x_next = x[1:]
    xmat = np.column_stack([np.ones_like(x_prev), x_prev])
    beta, *_ = np.linalg.lstsq(xmat, x_next, rcond=None)
    intercept = float(beta[0])
    phi = float(beta[1])
    fitted = intercept + phi * x_prev
    residuals = x_next - fitted
    residual_std = float(np.std(residuals, ddof=1))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((x_next - x_next.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return intercept, phi, residuals, residual_std, r2


def _ou_from_positive_phi(
    *,
    x: np.ndarray,
    intercept: float,
    phi: float,
    residual_std: float,
    r2: float,
    dt_years: float,
    method: str = "ou_ar1",
    warning: str = "",
) -> OUFit:
    kappa = -np.log(phi) / dt_years
    theta = intercept / (1.0 - phi)

    # Var(residual) = sigma^2 * (1 - exp(-2 kappa dt)) / (2 kappa)
    sigma = residual_std * np.sqrt(2.0 * kappa / (1.0 - phi**2))
    stationary_std = sigma / np.sqrt(2.0 * kappa)
    half_life_years = np.log(2.0) / kappa
    half_life_days = half_life_years * 252.0

    return OUFit(
        theta=float(theta),
        kappa=float(kappa),
        sigma=float(sigma),
        phi=float(phi),
        intercept=float(intercept),
        residual_std=float(residual_std),
        stationary_std=float(stationary_std),
        half_life_days=float(half_life_days),
        dt_years=float(dt_years),
        n_obs=int(len(x)),
        r2=float(r2),
        method=method,
        warning=warning,
    )


def fit_ou_ar1(
    values: pd.Series | np.ndarray | list[float],
    *,
    dt_years: float = 1 / 252,
    min_obs: int = 20,
) -> OUFit:
    """Fit OU parameters using the exact AR(1) discretization.

    X_next = intercept + phi * X_prev + residual
    phi = exp(-kappa * dt)
    theta = intercept / (1 - phi)

    This is the strict estimator: it requires 0 < phi < 1. If empirical data
    produce phi <= 0 or phi >= 1, use ``fit_ou_or_stationary_proxy`` for a
    plotting/scanning fallback that does not pretend the strict OU assumption held.
    """
    x = _prepare_values(values, min_obs=min_obs)
    intercept, phi, _residuals, residual_std, r2 = _ar1_ols(x)

    if not (0 < phi < 1):
        raise ValueError(
            f"Estimated AR(1) phi={phi:.4f}; OU mean reversion requires 0 < phi < 1."
        )

    return _ou_from_positive_phi(
        x=x,
        intercept=intercept,
        phi=phi,
        residual_std=residual_std,
        r2=r2,
        dt_years=dt_years,
    )


def fit_ou_or_stationary_proxy(
    values: pd.Series | np.ndarray | list[float],
    *,
    dt_years: float = 1 / 252,
    min_obs: int = 20,
    fallback_half_life_days: float = 20.0,
) -> OUFit:
    """Fit a strict OU model when possible, otherwise return a safe proxy.

    Real extracted skew series can be noisy. Sometimes the AR(1) slope is
    negative, meaning the discrete series is stationary but oscillatory. A
    continuous-time OU process cannot have negative phi, so strict fitting fails.

    This helper keeps the CLI usable:
    - if 0 < phi < 1: return the strict OU fit;
    - if -1 < phi < 0: use the stationary AR(1) mean/std, but simulate with
      a positive OU decay based on abs(phi). This is flagged as a proxy;
    - otherwise: use sample mean/std and a conservative fallback half-life,
      also flagged as a proxy.
    """
    x = _prepare_values(values, min_obs=min_obs)
    intercept, phi, _residuals, residual_std, r2 = _ar1_ols(x)

    if 0 < phi < 1:
        return _ou_from_positive_phi(
            x=x,
            intercept=intercept,
            phi=phi,
            residual_std=residual_std,
            r2=r2,
            dt_years=dt_years,
        )

    if -1 < phi < 0:
        theta = intercept / (1.0 - phi)
        stationary_std = residual_std / np.sqrt(max(1.0 - phi**2, 1e-12))
        phi_for_ou = max(abs(phi), 1e-6)
        kappa = -np.log(phi_for_ou) / dt_years
        sigma = stationary_std * np.sqrt(2.0 * kappa)
        half_life_days = np.log(2.0) / kappa * 252.0
        warning = (
            f"Strict OU rejected because AR(1) phi={phi:.4f} is negative. "
            "Using stationary AR(1) mean/std with abs(phi) as an OU-style simulation proxy."
        )
        return OUFit(
            theta=float(theta),
            kappa=float(kappa),
            sigma=float(sigma),
            phi=float(phi),
            intercept=float(intercept),
            residual_std=float(residual_std),
            stationary_std=float(stationary_std),
            half_life_days=float(half_life_days),
            dt_years=float(dt_years),
            n_obs=int(len(x)),
            r2=float(r2),
            method="stationary_ar1_abs_phi_proxy",
            warning=warning,
        )

    theta = float(np.mean(x))
    stationary_std = float(np.std(x, ddof=1))
    kappa = float(np.log(2.0) / (fallback_half_life_days / 252.0))
    phi_for_ou = float(np.exp(-kappa * dt_years))
    sigma = float(stationary_std * np.sqrt(2.0 * kappa)) if stationary_std > 0 else np.nan
    warning = (
        f"Strict OU rejected because AR(1) phi={phi:.4f} is not stationary/admissible. "
        f"Using sample mean/std with fallback half-life {fallback_half_life_days:g} trading days."
    )
    return OUFit(
        theta=theta,
        kappa=kappa,
        sigma=sigma,
        phi=phi,
        intercept=float(theta * (1.0 - phi_for_ou)),
        residual_std=float(residual_std),
        stationary_std=stationary_std,
        half_life_days=float(fallback_half_life_days),
        dt_years=float(dt_years),
        n_obs=int(len(x)),
        r2=float(r2),
        method="sample_moment_proxy",
        warning=warning,
    )


def infer_dt_years(dates: pd.Series, *, trading_days_per_year: int = 252) -> float:
    dt = pd.Series(pd.to_datetime(dates)).sort_values().diff().dt.days.dropna()
    if dt.empty:
        return 1 / trading_days_per_year
    median_calendar_days = max(float(dt.median()), 1.0)
    # For daily market data this maps one observation step to one trading day.
    # If the data is weekly, the median gap naturally increases the time step.
    return median_calendar_days / trading_days_per_year


def fit_ou_from_frame(
    frame: pd.DataFrame,
    *,
    value_col: str = "skew",
    date_col: str = "date",
    trading_days_per_year: int = 252,
    min_obs: int = 20,
) -> OUFit:
    clean = frame[[date_col, value_col]].dropna().sort_values(date_col)
    dt_years = infer_dt_years(clean[date_col], trading_days_per_year=trading_days_per_year)
    return fit_ou_ar1(clean[value_col], dt_years=dt_years, min_obs=min_obs)


def fit_ou_or_stationary_proxy_from_frame(
    frame: pd.DataFrame,
    *,
    value_col: str = "skew",
    date_col: str = "date",
    trading_days_per_year: int = 252,
    min_obs: int = 20,
    fallback_half_life_days: float = 20.0,
) -> OUFit:
    clean = frame[[date_col, value_col]].dropna().sort_values(date_col)
    dt_years = infer_dt_years(clean[date_col], trading_days_per_year=trading_days_per_year)
    return fit_ou_or_stationary_proxy(
        clean[value_col],
        dt_years=dt_years,
        min_obs=min_obs,
        fallback_half_life_days=fallback_half_life_days,
    )


def simulate_ou(
    *,
    theta: float,
    kappa: float,
    sigma: float,
    x0: float,
    n_steps: int,
    dt_years: float = 1 / 252,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_steps, dtype=float)
    out[0] = x0
    phi = np.exp(-kappa * dt_years)
    innovation_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt_years)) / (2.0 * kappa))
    for i in range(1, n_steps):
        mean = theta + phi * (out[i - 1] - theta)
        out[i] = mean + innovation_std * rng.normal()
    return out
