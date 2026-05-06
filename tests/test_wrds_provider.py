from __future__ import annotations

import pandas as pd

from ou_skew.data.wrds_provider import WRDSOptionMetricsProvider


def test_wrds_query_uses_2025_dates_and_ticker():
    provider = WRDSOptionMetricsProvider(connection=object())
    query = provider.build_query(tickers=["UBER"], start="2025-01-02", end="2025-12-31")

    assert "'UBER'" in query
    assert "2025-01-02" in query
    assert "2025-12-31" in query
    assert "optionm.opprcd2025" in query
    assert "optionm.secnmd" in query
    assert "optionm.secprd2025" in query
    assert "o.sec_price" not in query
    assert "s.close AS underlying_price" in query


class FakeConnection:
    def raw_sql(self, query: str, date_cols=None):
        assert "UBER" in query
        return pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-02"],
                "ticker": ["UBER", "UBER"],
                "expiry": ["2025-01-31", "2025-01-31"],
                "option_type": ["C", "P"],
                "strike": [60.0, 55.0],
                "underlying_price": [58.0, 58.0],
                "implied_volatility": [0.45, 0.55],
                "delta": [0.25, -0.25],
                "bid": [1.0, 1.1],
                "ask": [1.2, 1.3],
                "volume": [100, 100],
                "open_interest": [1000, 1000],
            }
        )


def test_wrds_provider_load_normalizes_fake_wrds_frame():
    provider = WRDSOptionMetricsProvider(connection=FakeConnection())
    out = provider.load(tickers=["UBER"], start="2025-01-02", end="2025-12-31")

    assert len(out) == 2
    assert set(out["option_type"]) == {"C", "P"}
    assert out["ticker"].iloc[0] == "UBER"
    assert "days_to_expiry" in out.columns


def test_wrds_table_config_can_override_tables():
    from ou_skew.data.wrds_provider import WRDSTableConfig

    provider = WRDSOptionMetricsProvider(
        connection=object(),
        tables=WRDSTableConfig(library="optionm_all", option_table="opprcd2025", price_table="secprd2025"),
    )
    query = provider.build_query(tickers=["UBER"], start="2025-01-02", end="2025-12-31")

    assert "optionm_all.opprcd2025" in query
    assert "optionm_all.secprd2025" in query


class ChunkFakeConnection:
    def __init__(self):
        self.calls = []

    def raw_sql(self, query: str, date_cols=None):
        self.calls.append(query)
        return pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-02"],
                "ticker": ["UBER", "UBER"],
                "expiry": ["2025-01-31", "2025-01-31"],
                "option_type": ["C", "P"],
                "strike": [60.0, 55.0],
                "underlying_price": [58.0, 58.0],
                "implied_volatility": [0.45, 0.55],
                "delta": [0.25, -0.25],
                "bid": [1.0, 1.1],
                "ask": [1.2, 1.3],
                "volume": [100, 100],
                "open_interest": [1000, 1000],
            }
        )


def test_wrds_provider_load_chunked_prints_multiple_queries():
    conn = ChunkFakeConnection()
    messages = []
    provider = WRDSOptionMetricsProvider(connection=conn)
    out = provider.load_chunked(
        tickers=["UBER"],
        start="2025-01-02",
        end="2025-02-10",
        chunk_days=20,
        progress=messages.append,
    )

    assert len(conn.calls) == 2
    assert len(out) == 4
    assert any("[1/2]" in msg for msg in messages)
    assert any("got 2 option rows" in msg for msg in messages)
