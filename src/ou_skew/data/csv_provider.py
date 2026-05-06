from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ou_skew.data.schema import normalize_option_chain


class CSVOptionDataProvider:
    """Load normalized option-chain data from CSV.

    This is the recommended provider for serious historical research because
    historical option chains usually come from WRDS OptionMetrics or another vendor.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(
        self,
        *,
        tickers: Iterable[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV not found: {self.path}")
        df = pd.read_csv(self.path)
        return normalize_option_chain(df, tickers=tickers, start=start, end=end)
