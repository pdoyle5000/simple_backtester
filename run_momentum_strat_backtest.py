from momentum_strategy.momentum_strategy import execute_momentum_strategy
from simple_backtester.backtester import BackTester
from copy import deepcopy
import pandas as pd
import json
import os
import pandas_market_calendars as pmc
import argparse


def get_valid_dates(start_date: str, end_date: str) -> pd.Series:
    valid = (
        pmc.get_calendar("NYSE")
        .schedule(start_date=start_date, end_date=end_date)
        .index
    )
    return valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--momentum", type=int, default=14)
    parser.add_argument("--volatility", type=int, default=14)
    parser.add_argument("--num_stocks", type=int, default=4)
    parser.add_argument("--data", type=str, default="data/nasdaq_backtest.csv")
    parser.add_argument("--output-label", default="default")
    return parser.parse_args()


if __name__ == "__main__":
    print("Importing data.")
    args = parse_args()
    all_daily = pd.read_csv(args.data)
    sp_daily = pd.read_csv("data/sp500_backtest.csv")
    for sym, df in sp_daily.groupby("symbol"):
        if len(all_daily.query("symbol == @sym")) == 0:
            all_daily = all_daily.append(df)
    all_daily["date"] = pd.to_datetime(all_daily.date)
    valid_days = all_daily[
        all_daily.date.isin(get_valid_dates("2016-01-01", "2021-02-12"))
    ]
    print(f"Csv imported. {len(valid_days)} records.")

    strat = execute_momentum_strategy(
        valid_days,
        momentum_window=args.momentum,
        volatility_window=args.volatility,
        num_stocks=args.num_stocks,
    )

    print("Strategy created: Starting backtest.")
    backtester = BackTester(strat, 10000.00, start_date="2017-01-01")
    os.makedirs("results", exist_ok=True)
    backtester.data.to_csv(f"results/{args.output_label}_backtest.csv", index=False)
    backtester.daily_totals.to_csv(
        f"results/{args.output_label}_totals.csv", index=False
    )
    backtester.ledger.to_csv(f"results/{args.output_label}_ledger.csv", index=False)
    with open(f"results/{args.output_label}_daily_state.json", "w") as fout:
        ds = {str(key): val for key, val in backtester.daily_state.items()}
        json.dump(ds, fout)
