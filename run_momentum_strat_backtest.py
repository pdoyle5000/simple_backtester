from momentum_strategy.momentum_strategy import execute_momentum_strategy
from simple_backtester.backtester import BackTester
import pandas as pd
import json
import os
import pandas_market_calendars as pmc


def get_valid_dates(start_date, end_date):
    valid = (
        pmc.get_calendar("NYSE")
        .schedule(start_date=start_date, end_date=end_date)
        .index
    )
    return valid


if __name__ == "__main__":
    print("Importing csv.")
    all_daily = pd.read_csv("data/nasdaq_backtest.csv")
    all_daily["date"] = pd.to_datetime(all_daily.date)
    valid_days = all_daily[
        all_daily.date.isin(get_valid_dates("2018-01-01", "2021-02-12"))
    ]
    print(f"Csv imported. {len(valid_days)} records.")

    strat = execute_momentum_strategy(
        valid_days, momentum_window=14, volatility_window=14, num_stocks=4
    )
    print("Strategy created: Starting backtest.")
    backtester = BackTester(strat, 10000.00)
    print("Backtest completed: Outputing results.")
    os.makedirs("results", exist_ok=True)
    backtester.data.to_csv("results/backtest.csv", index=False)
    backtester.daily_totals.to_csv("results/totals.csv", index=False)

    with open("results/daily_state.json", "w") as fout:
        ds = {str(key): val for key, val in backtester.daily_state.items()}
        json.dump(ds, fout)
