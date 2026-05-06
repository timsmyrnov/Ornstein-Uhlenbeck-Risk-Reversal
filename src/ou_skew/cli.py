from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ou_skew.backtest import rolling_ou_zscore_backtest
from ou_skew.config import load_config
from ou_skew.data.csv_provider import CSVOptionDataProvider
from ou_skew.data.yfinance_provider import YFinanceSnapshotProvider
from ou_skew.data.wrds_provider import WRDSOptionMetricsProvider, WRDSTableConfig, default_wrds_username
from ou_skew.features.skew import compute_skew_timeseries
from ou_skew.scan import scan_ou_skew
from ou_skew.zscores import rolling_ou_zscores


def _parse_tickers(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    tickers: list[str] = []
    for value in values:
        tickers.extend([x.strip().upper() for x in value.split(",") if x.strip()])
    return tickers or None


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def cmd_build_skew(args: argparse.Namespace) -> None:
    provider = CSVOptionDataProvider(args.csv)
    tickers = _parse_tickers(args.tickers)
    options = provider.load(tickers=tickers, start=args.start, end=args.end)
    skew = compute_skew_timeseries(
        options,
        tickers=tickers,
        target_tenor_days=args.tenor_days,
        call_delta=args.call_delta,
        put_delta=args.put_delta,
        skew_definition=args.skew_definition,
        min_volume=args.min_volume,
        min_open_interest=args.min_open_interest,
        max_spread_pct=args.max_spread_pct,
    )
    _ensure_parent(args.out)
    skew.to_csv(args.out, index=False)
    print(f"Wrote {len(skew):,} skew observations to {args.out}")


def cmd_scan(args: argparse.Namespace) -> None:
    provider = CSVOptionDataProvider(args.csv)
    tickers = _parse_tickers(args.tickers)
    options = provider.load(tickers=tickers, start=args.start, end=args.end)
    report = scan_ou_skew(
        options,
        tickers=tickers,
        lookback_days=args.lookback_days,
        target_tenor_days=args.tenor_days,
        call_delta=args.call_delta,
        put_delta=args.put_delta,
        skew_definition=args.skew_definition,
        min_volume=args.min_volume,
        min_open_interest=args.min_open_interest,
        max_spread_pct=args.max_spread_pct,
        forecast_horizon_days=args.forecast_horizon_days,
        min_obs=args.min_obs,
    )
    _ensure_parent(args.out)
    report.to_csv(args.out, index=False)
    print(report.to_string(index=False))
    print(f"\nWrote scan report to {args.out}")


def cmd_scan_config(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    provider = CSVOptionDataProvider(cfg.input_csv)
    options = provider.load(tickers=cfg.tickers)
    report = scan_ou_skew(
        options,
        tickers=cfg.tickers,
        lookback_days=cfg.ou.lookback_days,
        target_tenor_days=cfg.skew.target_tenor_days,
        call_delta=cfg.skew.call_delta,
        put_delta=cfg.skew.put_delta,
        skew_definition=cfg.skew.skew_definition,
        min_volume=cfg.skew.min_volume,
        min_open_interest=cfg.skew.min_open_interest,
        trading_days_per_year=cfg.ou.trading_days_per_year,
        forecast_horizon_days=cfg.ou.forecast_horizon_days,
    )
    out = Path(cfg.output_dir) / "scan.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(report.to_string(index=False))
    print(f"\nWrote scan report to {out}")


def cmd_backtest(args: argparse.Namespace) -> None:
    skew = pd.read_csv(args.skew_csv, parse_dates=["date"])
    result = rolling_ou_zscore_backtest(
        skew,
        ticker=args.ticker,
        window=args.window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        min_obs=args.min_obs,
    )
    _ensure_parent(args.out)
    result.to_csv(args.out, index=False)
    print(f"Wrote backtest to {args.out}")
    if not result.empty:
        print(result.tail(10).to_string(index=False))


def cmd_plot(args: argparse.Namespace) -> None:
    from ou_skew.plotting import plot_skew_with_ou_mean

    skew = pd.read_csv(args.skew_csv, parse_dates=["date"])
    out = plot_skew_with_ou_mean(skew, ticker=args.ticker, out_path=args.out, min_obs=args.min_obs)
    print(f"Wrote plot to {out}")


def cmd_simulate(args: argparse.Namespace) -> None:
    from ou_skew.plotting import (
        plot_ou_simulated_paths,
        plot_ou_terminal_distribution,
        simulate_ou_paths_from_skew,
    )

    skew = pd.read_csv(args.skew_csv, parse_dates=["date"])
    paths, fit = simulate_ou_paths_from_skew(
        skew,
        ticker=args.ticker,
        n_steps=args.steps,
        n_paths=args.paths,
        lookback_days=args.lookback_days,
        min_obs=args.min_obs,
        seed=args.seed,
    )

    out = plot_ou_simulated_paths(
        paths,
        fit=fit,
        ticker=args.ticker,
        out_path=args.out,
        max_paths_to_draw=args.max_paths_to_draw,
    )
    print(f"Wrote simulated path plot to {out}")

    if args.terminal_out:
        terminal_out = plot_ou_terminal_distribution(
            paths,
            fit=fit,
            ticker=args.ticker,
            out_path=args.terminal_out,
            bins=args.bins,
        )
        print(f"Wrote terminal distribution plot to {terminal_out}")

    if args.paths_out:
        _ensure_parent(args.paths_out)
        paths.to_csv(args.paths_out, index=False)
        print(f"Wrote simulated path data to {args.paths_out}")

    print(
        "Fit: "
        f"method={fit.method}, theta={fit.theta:.6g}, kappa={fit.kappa:.6g}, sigma={fit.sigma:.6g}, "
        f"phi={fit.phi:.6g}, half_life_days={fit.half_life_days:.3g}"
    )
    if fit.warning:
        print(f"Warning: {fit.warning}")


def cmd_zscores(args: argparse.Namespace) -> None:
    from ou_skew.plotting import plot_rolling_zscores

    skew = pd.read_csv(args.skew_csv, parse_dates=["date"])
    zscores = rolling_ou_zscores(
        skew,
        ticker=args.ticker,
        window=args.window,
        min_obs=args.min_obs,
        forecast_horizon_days=args.forecast_horizon_days,
    )
    _ensure_parent(args.out)
    zscores.to_csv(args.out, index=False)
    print(f"Wrote {len(zscores):,} rolling z-score rows to {args.out}")
    if args.plot_out:
        plot_out = plot_rolling_zscores(zscores, ticker=args.ticker, out_path=args.plot_out)
        print(f"Wrote z-score plot to {plot_out}")
    if not zscores.empty:
        cols = ["date", "ticker", "skew", "theta", "z_score", "expected_change_horizon", "status"]
        print(zscores[cols].tail(10).to_string(index=False))


def cmd_yfinance_snapshot(args: argparse.Namespace) -> None:
    tickers = _parse_tickers(args.tickers)
    if not tickers:
        raise ValueError("Provide at least one ticker.")
    provider = YFinanceSnapshotProvider(risk_free_rate=args.risk_free_rate)
    df = provider.load(tickers=tickers)
    _ensure_parent(args.out)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} option rows to {args.out}")


def _wrds_tables_from_args(args: argparse.Namespace) -> WRDSTableConfig:
    return WRDSTableConfig(
        library=args.library,
        option_table=args.option_table,
        name_table=args.name_table,
        price_table=args.price_table,
    )


def _wrds_username_from_args(args: argparse.Namespace) -> str | None:
    if args.wrds_username:
        return args.wrds_username
    default = default_wrds_username()
    prompt = f"WRDS username [{default}]: " if default else "WRDS username: "
    typed = input(prompt).strip()
    return typed or default or None


def cmd_wrds_fetch(args: argparse.Namespace) -> None:
    tickers = _parse_tickers(args.tickers)
    if not tickers:
        raise ValueError("Provide at least one ticker.")
    provider = WRDSOptionMetricsProvider(
        username=_wrds_username_from_args(args),
        tables=_wrds_tables_from_args(args),
    )
    out = provider.fetch_to_csv(
        tickers=tickers,
        start=args.start,
        end=args.end,
        out=args.out,
        chunk_days=args.chunk_days,
        progress=print,
    )
    print(f"Wrote WRDS OptionMetrics option-chain cache to {out}")


def cmd_wrds_skew(args: argparse.Namespace) -> None:
    tickers = _parse_tickers(args.tickers)
    if not tickers:
        raise ValueError("Provide at least one ticker.")
    provider = WRDSOptionMetricsProvider(
        username=_wrds_username_from_args(args),
        tables=_wrds_tables_from_args(args),
    )
    print("Fetching WRDS OptionMetrics data in chunks. This can still take a while, but progress should print below.")
    if args.chunk_days > 0:
        options = provider.load_chunked(
            tickers=tickers,
            start=args.start,
            end=args.end,
            chunk_days=args.chunk_days,
            progress=print,
        )
    else:
        print("Chunking disabled; running one full-range WRDS query ...")
        options = provider.load(tickers=tickers, start=args.start, end=args.end)
    print(f"Loaded {len(options):,} normalized option rows from WRDS.")
    if args.raw_out:
        _ensure_parent(args.raw_out)
        options.to_csv(args.raw_out, index=False)
        print(f"Wrote WRDS raw option-chain cache to {args.raw_out}")
    skew = compute_skew_timeseries(
        options,
        tickers=tickers,
        target_tenor_days=args.tenor_days,
        call_delta=args.call_delta,
        put_delta=args.put_delta,
        skew_definition=args.skew_definition,
        min_volume=args.min_volume,
        min_open_interest=args.min_open_interest,
        max_spread_pct=args.max_spread_pct,
    )
    _ensure_parent(args.out)
    skew.to_csv(args.out, index=False)
    print(f"Wrote {len(skew):,} WRDS-derived skew observations to {args.out}")
    if not skew.empty:
        print(skew.tail(10).to_string(index=False))


def cmd_wrds_info(args: argparse.Namespace) -> None:
    provider = WRDSOptionMetricsProvider(
        username=_wrds_username_from_args(args),
        tables=_wrds_tables_from_args(args),
    )
    start = getattr(args, "start", "2025-01-02")
    end = getattr(args, "end", "2025-12-31")
    tables = provider.resolved_tables(start=start, end=end)
    rows = []
    for table in [tables.option_ref, tables.name_ref, tables.price_ref]:
        lib, tbl = table.split(".", 1)
        try:
            desc = provider.connection.describe_table(lib, tbl)
            col = "name" if "name" in desc.columns else desc.columns[0]
            rows.append({"table": table, "columns": ", ".join(map(str, desc[col].head(60)))})
        except Exception as exc:  # pragma: no cover
            rows.append({"table": table, "columns": f"ERROR: {exc}"})
    print(pd.DataFrame(rows).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ou-skew",
        description="OU mean-reversion scanner for equity option implied-volatility skew.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common_option_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--csv", required=True, help="Input option-chain CSV.")
        p.add_argument("--tickers", nargs="*", help="Tickers, either space-separated or comma-separated.")
        p.add_argument("--start", default=None, help="Optional start date, YYYY-MM-DD.")
        p.add_argument("--end", default=None, help="Optional end date, YYYY-MM-DD.")
        p.add_argument("--tenor-days", type=int, default=30)
        p.add_argument("--call-delta", type=float, default=0.25)
        p.add_argument("--put-delta", type=float, default=-0.25)
        p.add_argument(
            "--skew-definition",
            choices=["put_minus_call", "call_minus_put", "put_minus_atm"],
            default="put_minus_call",
        )
        p.add_argument("--min-volume", type=int, default=None)
        p.add_argument("--min-open-interest", type=int, default=None)
        p.add_argument("--max-spread-pct", type=float, default=None)

    def add_wrds_connection_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--wrds-username", default=None, help="WRDS username. If omitted, the CLI prompts.")
        p.add_argument("--library", default="optionm", help="WRDS OptionMetrics library/schema name.")
        p.add_argument("--option-table", default="auto", help="OptionMetrics option-price table/view. Default auto resolves to opprcdYYYY from --start.")
        p.add_argument("--name-table", default="secnmd", help="OptionMetrics security-name table/view.")
        p.add_argument("--price-table", default="auto", help="OptionMetrics underlying-price table/view. Default auto resolves to secprdYYYY from --start.")
        p.add_argument("--chunk-days", type=int, default=31, help="WRDS date chunk size. Use 0 to disable chunking.")

    p = sub.add_parser("build-skew", help="Extract skew features from option-chain CSV.")
    add_common_option_args(p)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_build_skew)

    p = sub.add_parser("scan", help="Extract skew and fit OU model ticker-by-ticker.")
    add_common_option_args(p)
    p.add_argument("--lookback-days", type=int, default=252)
    p.add_argument("--forecast-horizon-days", type=int, default=30)
    p.add_argument("--min-obs", type=int, default=20)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("scan-config", help="Run a scan from YAML config.")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_scan_config)

    p = sub.add_parser("backtest", help="Rolling OU z-score signal backtest on a skew CSV.")
    p.add_argument("--skew-csv", required=True)
    p.add_argument("--ticker", required=True)
    p.add_argument("--window", type=int, default=126)
    p.add_argument("--entry-z", type=float, default=1.5)
    p.add_argument("--exit-z", type=float, default=0.25)
    p.add_argument("--min-obs", type=int, default=40)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("plot", help="Plot skew and fitted OU mean.")
    p.add_argument("--skew-csv", required=True)
    p.add_argument("--ticker", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-obs", type=int, default=20)
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("simulate", help="Simulate future OU skew paths and write plot images.")
    p.add_argument("--skew-csv", required=True)
    p.add_argument("--ticker", required=True)
    p.add_argument("--steps", type=int, default=63, help="Number of simulated business-day steps.")
    p.add_argument("--paths", type=int, default=250, help="Number of simulated paths.")
    p.add_argument("--lookback-days", type=int, default=None)
    p.add_argument("--min-obs", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True, help="Output PNG for simulated paths.")
    p.add_argument("--terminal-out", default=None, help="Optional terminal distribution PNG.")
    p.add_argument("--paths-out", default=None, help="Optional CSV of simulated path values.")
    p.add_argument("--max-paths-to-draw", type=int, default=250)
    p.add_argument("--bins", type=int, default=60)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("zscores", help="Generate rolling OU z-score CSV and optional z-score plot.")
    p.add_argument("--skew-csv", required=True)
    p.add_argument("--ticker", required=True)
    p.add_argument("--window", type=int, default=126)
    p.add_argument("--forecast-horizon-days", type=int, default=30)
    p.add_argument("--min-obs", type=int, default=40)
    p.add_argument("--out", required=True)
    p.add_argument("--plot-out", default=None)
    p.set_defaults(func=cmd_zscores)

    p = sub.add_parser("wrds-fetch", help="Fetch historical OptionMetrics option chains from WRDS into a CSV cache.")
    add_wrds_connection_args(p)
    p.add_argument("--tickers", nargs="*", required=True)
    p.add_argument("--start", default="2025-01-02", help="Start date, YYYY-MM-DD. Default is a 2025 window.")
    p.add_argument("--end", default="2025-12-31", help="End date, YYYY-MM-DD. Default is a 2025 window.")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_wrds_fetch)

    p = sub.add_parser("wrds-skew", help="Fetch WRDS OptionMetrics data and directly build a skew CSV.")
    add_wrds_connection_args(p)
    p.add_argument("--tickers", nargs="*", required=True)
    p.add_argument("--start", default="2025-01-02", help="Start date, YYYY-MM-DD. Default is a 2025 window.")
    p.add_argument("--end", default="2025-12-31", help="End date, YYYY-MM-DD. Default is a 2025 window.")
    p.add_argument("--tenor-days", type=int, default=30)
    p.add_argument("--call-delta", type=float, default=0.25)
    p.add_argument("--put-delta", type=float, default=-0.25)
    p.add_argument("--skew-definition", choices=["put_minus_call", "call_minus_put", "put_minus_atm"], default="put_minus_call")
    p.add_argument("--min-volume", type=int, default=None)
    p.add_argument("--min-open-interest", type=int, default=None)
    p.add_argument("--max-spread-pct", type=float, default=None)
    p.add_argument("--raw-out", default=None, help="Optional normalized WRDS option-chain CSV cache.")
    p.add_argument("--out", required=True, help="Output skew CSV.")
    p.set_defaults(func=cmd_wrds_skew)

    p = sub.add_parser("wrds-info", help="Inspect configured WRDS OptionMetrics table columns.")
    add_wrds_connection_args(p)
    p.add_argument("--start", default="2025-01-02", help="Date used to resolve auto table names.")
    p.add_argument("--end", default="2025-12-31", help="Date used to resolve auto table names.")
    p.set_defaults(func=cmd_wrds_info)

    p = sub.add_parser("yfinance-snapshot", help="Download current option-chain snapshots.")
    p.add_argument("--tickers", nargs="*", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--risk-free-rate", type=float, default=0.04)
    p.set_defaults(func=cmd_yfinance_snapshot)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
