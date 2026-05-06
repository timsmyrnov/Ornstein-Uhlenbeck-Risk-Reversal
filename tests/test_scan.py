from pathlib import Path

from ou_skew.data.csv_provider import CSVOptionDataProvider
from ou_skew.scan import scan_ou_skew


def test_scan_on_sample_data():
    root = Path(__file__).resolve().parents[1]
    provider = CSVOptionDataProvider(root / "data" / "sample_option_chains.csv")
    options = provider.load(tickers=["AAPL"])
    report = scan_ou_skew(options, tickers=["AAPL"], min_obs=20)

    assert len(report) == 1
    assert report.loc[0, "ticker"] == "AAPL"
    assert report.loc[0, "status"] in {"ou_ar1", "stationary_ar1_abs_phi_proxy", "sample_moment_proxy"}
