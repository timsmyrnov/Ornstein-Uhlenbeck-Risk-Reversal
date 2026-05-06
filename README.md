# OU Skew Scanner

A Python research framework for modeling mean reversion in equity-option implied-volatility skew features using an Ornstein-Uhlenbeck (OU) process.

The project is intentionally built around **skew features**, not the entire volatility surface. A typical feature is:

```text
skew_t = IV_25Δ_put,t - IV_25Δ_call,t
```

Then the feature is modeled as:

```text
dX_t = kappa * (theta - X_t) dt + sigma dW_t
```

where:

- `theta` is the long-run mean skew,
- `kappa` is the mean-reversion speed,
- `sigma` is the OU diffusion volatility,
- `half_life_days = log(2) / kappa`, converted into trading days.

## Why this repo exists

The main question is:

> Is current option skew unusually rich or cheap relative to its own mean-reverting history?

The scanner estimates OU parameters ticker-by-ticker and ranks current skew dislocations by z-score, half-life, and expected reversion.

## Important data note

Historical option-chain data is not free in the same way historical stock prices are. This repo therefore uses a **CSV provider as the main research path**. It works well with OptionMetrics/WRDS-style exports or any vendor data normalized to the schema below.

A yfinance snapshot provider is included for convenience, but it only gives current listed chains. It is useful for collecting daily snapshots going forward, not for reconstructing deep historical skew by itself.

## Input CSV schema

Required columns:

| column | meaning |
|---|---|
| `date` | observation date |
| `ticker` | underlying ticker |
| `expiry` | option expiration date |
| `option_type` | `C`/`P` or `call`/`put` |
| `strike` | option strike |
| `underlying_price` | spot/underlying price on `date` |
| `implied_volatility` | IV as decimal, e.g. `0.32`; percent values like `32` are auto-normalized |
| `delta` | option delta; calls positive, puts negative |

Optional liquidity columns:

| column | meaning |
|---|---|
| `bid` | option bid |
| `ask` | option ask |
| `volume` | daily volume |
| `open_interest` | open interest |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,yfinance]"
```

Or without yfinance:

```bash
pip install -e ".[dev]"
```

## Quick start with sample data

Build a 30-day 25-delta put-minus-call skew series:

```bash
ou-skew build-skew \
  --csv data/sample_option_chains.csv \
  --tickers AAPL MSFT \
  --tenor-days 30 \
  --out reports/skew.csv
```

Run the OU scanner:

```bash
ou-skew scan \
  --csv data/sample_option_chains.csv \
  --tickers AAPL MSFT \
  --lookback-days 252 \
  --tenor-days 30 \
  --out reports/scan.csv
```

Plot one ticker's skew series:

```bash
ou-skew plot \
  --skew-csv reports/skew.csv \
  --ticker AAPL \
  --out reports/aapl_skew.png
```

Run a simple rolling z-score mean-reversion backtest:

```bash
ou-skew backtest \
  --skew-csv reports/skew.csv \
  --ticker AAPL \
  --window 60 \
  --entry-z 1.5 \
  --exit-z 0.25 \
  --out reports/aapl_backtest.csv
```

Generate rolling OU z-scores and a z-score chart:

```bash
ou-skew zscores \
  --skew-csv reports/skew.csv \
  --ticker AAPL \
  --window 126 \
  --out reports/aapl_zscores.csv \
  --plot-out reports/aapl_zscores.png
```

Simulate future OU skew paths and terminal skew distribution:

```bash
ou-skew simulate \
  --skew-csv reports/skew.csv \
  --ticker AAPL \
  --steps 63 \
  --paths 500 \
  --out reports/aapl_ou_paths.png \
  --terminal-out reports/aapl_ou_terminal.png \
  --paths-out reports/aapl_ou_paths.csv
```


## CLI commands

### `build-skew`

Extracts a skew feature from option chains.

```bash
ou-skew build-skew --csv path/to/options.csv --tickers AAPL MSFT --out reports/skew.csv
```

### `scan`

Extracts skew and fits an OU process to each ticker.

```bash
ou-skew scan --csv path/to/options.csv --tickers AAPL MSFT --out reports/scan.csv
```

### `backtest`

Runs a simple mean-reversion signal using rolling OU z-scores.

```bash
ou-skew backtest --skew-csv reports/skew.csv --ticker AAPL --out reports/backtest.csv
```



### `simulate`

Fits an OU process to a ticker's skew history, simulates future skew paths, and writes PNG images.

```bash
ou-skew simulate --skew-csv reports/skew.csv --ticker AAPL --out reports/aapl_ou_paths.png --terminal-out reports/aapl_ou_terminal.png
```

Useful options:

| option | meaning |
|---|---|
| `--steps` | number of future business-day steps |
| `--paths` | number of Monte Carlo paths |
| `--lookback-days` | fit OU only on a recent lookback window |
| `--paths-out` | optional CSV of simulated path values |

### `zscores`

Generates rolling OU z-scores from an existing skew CSV.

```bash
ou-skew zscores --skew-csv reports/skew.csv --ticker AAPL --out reports/aapl_zscores.csv --plot-out reports/aapl_zscores.png
```

### `yfinance-snapshot`

Downloads current option chains. This is optional and requires the `yfinance` extra.

```bash
ou-skew yfinance-snapshot --tickers AAPL MSFT --out data/snapshots/options_today.csv
```

## Python usage

```python
from ou_skew.data.csv_provider import CSVOptionDataProvider
from ou_skew.features.skew import compute_skew_timeseries
from ou_skew.scan import scan_ou_skew

provider = CSVOptionDataProvider("data/sample_option_chains.csv")
options = provider.load(tickers=["AAPL", "MSFT"])

skew = compute_skew_timeseries(
    options,
    tickers=["AAPL", "MSFT"],
    target_tenor_days=30,
    call_delta=0.25,
    put_delta=-0.25,
    skew_definition="put_minus_call",
)

report = scan_ou_skew(
    options,
    tickers=["AAPL", "MSFT"],
    lookback_days=252,
    target_tenor_days=30,
)

print(report)
```

## Model details

The OU process is estimated through the exact AR(1) discretization under evenly spaced observations:

```text
X_{t+dt} = alpha + phi X_t + epsilon_t
phi = exp(-kappa dt)
theta = alpha / (1 - phi)
kappa = -log(phi) / dt
```

The stationary standard deviation is:

```text
sigma / sqrt(2 * kappa)
```

The current dislocation score is:

```text
z = (latest_skew - theta) / stationary_std
```

## Project layout

```text
ou-skew-scanner/
  src/ou_skew/
    data/          # data providers and schema normalization
    features/      # skew extraction logic
    models/        # OU calibration and simulation
    backtest.py    # rolling signal backtest
    cli.py         # command line interface
    plotting.py    # chart helpers, future OU path plots, z-score plots
    scan.py        # ticker scanner
    zscores.py     # rolling OU z-score generation
  tests/           # unit tests
  configs/         # example YAML config
  data/            # sample synthetic option-chain data
```

## Limitations

- OU is a simple stationary model. Equity skew can jump or shift regimes around earnings, crashes, borrow stress, liquidity shocks, and macro events.
- The model is more defensible for a fixed-tenor scalar feature than for the entire smile.
- Backtests here are signal research backtests, not execution-ready PnL simulations. Real skew trades require bid/ask, margin, carry, vega normalization, and hedge mechanics.

## Suggested next extensions

- Add WRDS OptionMetrics query integration.
- Add earnings-date filters.
- Add liquidity filters by bid/ask spread and open interest.
- Add multiple tenors: 30D, 60D, 90D.
- Fit a vector OU model to `level`, `slope`, and `curvature` smile factors.
- Add SVI/SABR smile fitting and model the fitted skew parameter.

## WRDS / OptionMetrics workflow

This repo can pull historical OptionMetrics IvyDB data directly from WRDS, then cache the normalized option-chain rows locally as CSV. That cache is what the plotting, simulation, and z-score commands use afterward.

Install the WRDS extra:

```bash
python3 -m pip install -e ".[dev,wrds]"
```

Fetch WRDS data and directly build a skew file. The defaults are intentionally a 2025 date window:

```bash
ou-skew wrds-skew \
  --tickers UBER \
  --start 2025-01-02 \
  --end 2025-12-31 \
  --tenor-days 30 \
  --raw-out data/uber_optionmetrics_2025.csv \
  --out reports/uber_skew_2025.csv
```

If `--wrds-username` is omitted, the CLI asks for your WRDS username and the `wrds` package handles the password/2FA login.

Then plot/simulate/score:

```bash
ou-skew plot \
  --skew-csv reports/uber_skew_2025.csv \
  --ticker UBER \
  --out reports/uber_skew_ou_mean.png

ou-skew simulate \
  --skew-csv reports/uber_skew_2025.csv \
  --ticker UBER \
  --steps 63 \
  --paths 500 \
  --out reports/uber_ou_paths.png \
  --terminal-out reports/uber_ou_terminal.png \
  --paths-out reports/uber_ou_paths.csv

ou-skew zscores \
  --skew-csv reports/uber_skew_2025.csv \
  --ticker UBER \
  --window 126 \
  --min-obs 40 \
  --out reports/uber_zscores.csv \
  --plot-out reports/uber_zscores.png
```

If your WRDS institution exposes different OptionMetrics table/view names, inspect the columns:

```bash
ou-skew wrds-info
```

The default table assumptions are:

- option table: `optionm.opprcd2025`
- security-name table: `optionm.secnmd`
- underlying-price table: `optionm.secprd2025`

Override them with `--library`, `--option-table`, `--name-table`, or `--price-table` when needed.

### WRDS 2025 table note

For 2025 OptionMetrics data, the CLI now defaults to year-specific WRDS tables:

- `optionm.opprcd2025` for option prices
- `optionm.secprd2025` for underlying prices
- `optionm.secnmd` for ticker/security names

So the plain `wrds-skew` command should work for a 2025 date range without table overrides.

## Note on non-admissible OU fits

The strict OU calibration requires the estimated AR(1) slope to satisfy `0 < phi < 1`.
Real extracted skew series can be noisy; if the slope is negative or otherwise not
OU-admissible, plotting/simulation/z-score commands now fall back to a clearly
flagged stationary proxy instead of crashing. The output `status`/`method` field
will show whether the result is a strict `ou_ar1` fit or a proxy fit.
