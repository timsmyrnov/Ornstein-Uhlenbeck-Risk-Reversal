from pathlib import Path

import pandas as pd

from ou_skew.data.csv_provider import CSVOptionDataProvider
from ou_skew.features.skew import compute_skew_timeseries
from ou_skew.plotting import (
    plot_ou_simulated_paths,
    plot_ou_terminal_distribution,
    plot_rolling_zscores,
    simulate_ou_paths_from_skew,
)
from ou_skew.zscores import rolling_ou_zscores


def _sample_skew() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    provider = CSVOptionDataProvider(root / "data" / "sample_option_chains.csv")
    options = provider.load(tickers=["AAPL"])
    return compute_skew_timeseries(options, tickers=["AAPL"])


def test_rolling_ou_zscores_generates_rows():
    skew = _sample_skew()
    z = rolling_ou_zscores(skew, ticker="AAPL", window=60, min_obs=20)

    assert not z.empty
    assert {"date", "ticker", "z_score", "theta", "expected_change_horizon"}.issubset(z.columns)


def test_future_simulation_plotting(tmp_path):
    skew = _sample_skew()
    paths, fit = simulate_ou_paths_from_skew(
        skew, ticker="AAPL", n_steps=10, n_paths=5, min_obs=20, seed=1
    )

    assert len(paths) == 50
    assert fit.kappa > 0

    paths_png = plot_ou_simulated_paths(paths, fit=fit, ticker="AAPL", out_path=tmp_path / "paths.png")
    terminal_png = plot_ou_terminal_distribution(
        paths, fit=fit, ticker="AAPL", out_path=tmp_path / "terminal.png"
    )

    assert paths_png.exists()
    assert terminal_png.exists()


def test_zscore_plotting(tmp_path):
    skew = _sample_skew()
    z = rolling_ou_zscores(skew, ticker="AAPL", window=60, min_obs=20)
    out = plot_rolling_zscores(z, ticker="AAPL", out_path=tmp_path / "zscores.png")

    assert out.exists()
