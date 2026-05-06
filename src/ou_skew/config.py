from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SkewConfig:
    target_tenor_days: int = 30
    call_delta: float = 0.25
    put_delta: float = -0.25
    skew_definition: str = "put_minus_call"
    min_volume: int | None = None
    min_open_interest: int | None = None


@dataclass(frozen=True)
class OUConfig:
    lookback_days: int = 252
    trading_days_per_year: int = 252
    forecast_horizon_days: int = 30


@dataclass(frozen=True)
class AppConfig:
    input_csv: str
    output_dir: str = "reports"
    tickers: list[str] = field(default_factory=list)
    skew: SkewConfig = field(default_factory=SkewConfig)
    ou: OUConfig = field(default_factory=OUConfig)


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return AppConfig(
        input_csv=raw["input_csv"],
        output_dir=raw.get("output_dir", "reports"),
        tickers=list(raw.get("tickers", [])),
        skew=SkewConfig(**raw.get("skew", {})),
        ou=OUConfig(**raw.get("ou", {})),
    )
