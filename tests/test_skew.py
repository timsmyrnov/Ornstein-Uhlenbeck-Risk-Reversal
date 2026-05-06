import pandas as pd

from ou_skew.data.schema import normalize_option_chain
from ou_skew.features.skew import compute_skew_timeseries, select_nearest_contract


def _sample_options():
    return normalize_option_chain(
        pd.DataFrame(
            [
                {
                    "date": "2026-01-02",
                    "ticker": "ABC",
                    "expiry": "2026-02-01",
                    "option_type": "C",
                    "strike": 105,
                    "underlying_price": 100,
                    "implied_volatility": 0.22,
                    "delta": 0.25,
                    "volume": 10,
                    "open_interest": 100,
                },
                {
                    "date": "2026-01-02",
                    "ticker": "ABC",
                    "expiry": "2026-02-01",
                    "option_type": "P",
                    "strike": 95,
                    "underlying_price": 100,
                    "implied_volatility": 0.31,
                    "delta": -0.25,
                    "volume": 10,
                    "open_interest": 100,
                },
            ]
        )
    )


def test_select_nearest_contract():
    df = _sample_options()
    c = select_nearest_contract(
        df, option_type="C", target_delta=0.25, target_tenor_days=30
    )
    assert c is not None
    assert c.implied_volatility == 0.22


def test_compute_skew_timeseries_put_minus_call():
    df = _sample_options()
    skew = compute_skew_timeseries(df, tickers=["ABC"])
    assert len(skew) == 1
    assert abs(float(skew.loc[0, "skew"]) - 0.09) < 1e-12
