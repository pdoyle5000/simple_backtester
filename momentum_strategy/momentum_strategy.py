import pandas as pd
from typing import Union, Callable
from simple_backtester.backtester import Action
import numpy as np
from scipy import stats


def _momentum_score(ts: Union[pd.Series, np.ndarray]) -> float:
    # something to tweak here, the num of trading days
    if ts.min() <= 0:
        print("Cannot Compute Momentum, timeseries contains value(s) =< 0.")
        return -10
    trading_days = 252
    slope, _, r_value = stats.linregress(np.arange(len(ts)), np.log(ts))[:3]
    annualized_slope = (np.power(np.exp(slope), trading_days) - 1) * 100
    return round(annualized_slope * (r_value ** 2), 3)


def _inv_volatility(ts: pd.Series) -> float:
    vol = ts.pct_change(fill_method="ffill").std()
    return 1 / vol


def _rolling_groupby(df: pd.DataFrame, roll_func: Callable, window: int) -> pd.Series:
    return (
        df.groupby("symbol")["close"]
        .rolling(window)
        .apply(roll_func)
        .reset_index()
        .set_index("level_1")["close"]
    )


def _apply_actions(
    df: pd.DataFrame, num_stocks: int, drawdown_threshold: float = 0.2
) -> pd.DataFrame:
    # ASSUME DF is in ascending date order.
    df = df.dropna(subset=["momentum", "inv_volatility"])

    action_df = pd.DataFrame()
    for i, daily_df in df.groupby("date"):
        if i == df.date.min():  # its the first day!
            sorted_df = (
                daily_df.sort_values(by="momentum", ascending=False)
                .reset_index(drop=True)
                .loc[: num_stocks - 1]
            )
            sorted_df["action"] = Action.buy

            # day 1 drawdown protection
            sorted_df.loc[sorted_df.momentum < drawdown_threshold, "action"] = np.nan
            action_df = pd.concat([action_df, sorted_df])

        else:  # a previous day exists.
            sorted_df = (
                daily_df.sort_values(by="momentum", ascending=False)
                .reset_index(drop=True)
                .loc[: num_stocks - 1]
            )
            previous_day = action_df.query(
                "date == date.max() and action != @Action.sell and action"
            )

            # buys and holds
            for j, row in sorted_df.iterrows():
                if row.symbol in previous_day.symbol.values:
                    sorted_df.at[j, "action"] = Action.hold
                else:
                    sorted_df.at[j, "action"] = Action.buy

            # Drawdown Protection:  revert holds and buys if needed.
            sorted_df.at[
                (sorted_df.momentum < drawdown_threshold)
                & (sorted_df.action == Action.hold),
                "action",
            ] = Action.sell

            sorted_df.at[
                (sorted_df.momentum < drawdown_threshold)
                & (sorted_df.action == Action.buy),
                "action",
            ] = np.nan

            # Sells

            for _, row in previous_day.iterrows():
                if row.symbol not in sorted_df.symbol.values:
                    sell_row = pd.DataFrame(
                        [
                            {
                                "symbol": row.symbol,
                                "action": Action.sell,
                                "close": row.close,
                                "momentum": row.momentum,
                                "inv_volatility": row.inv_volatility,
                                "date": sorted_df.at[0, "date"],
                            }
                        ]
                    )
                    action_df = action_df.append(sell_row, ignore_index=True)
            action_df = pd.concat([action_df, sorted_df]).reset_index(drop=True)
    return action_df


def _apply_weights(action_df: pd.DataFrame) -> pd.Series:
    return (
        action_df.query("action != @Action.sell")
        .groupby("date")["inv_volatility"]
        .apply(lambda x: x / x.sum())
    )


def execute_momentum_strategy(
    df: pd.DataFrame,
    momentum_window: int = 30,
    volatility_window: int = 20,
    num_stocks: int = 2,
):
    """
    This strategy uses simple momentum to make stock transactions.

    This algorithm will run daily.  We will choose the top N
    stocks momentum-wise and rebalance every time.  This class takes a DataFrame
    of prices, dates and symbols and adds Actions and ownership weights.
    """
    # Thread out on the groupbys.. maybe a different workflow.
    # Group by to calc momentum and volatility all at once
    # (thread out the groups by symbol)
    # then apply actions. (cant be threaded.)
    # then thread out to apply weights per day.

    # assert that date is a datetime dtype.
    if df.date.dtype != np.dtype("datetime64[ns]"):
        print("date column needs to be datetime type.")
        raise ValueError

    df = df.sort_values(by="date").reset_index(drop=True)
    print("Calculating Momentum")
    df["momentum"] = _rolling_groupby(df, _momentum_score, momentum_window)
    print("Calculating Inverse Volatility.")
    df["inv_volatility"] = _rolling_groupby(df, _inv_volatility, volatility_window)
    df.dropna(subset=["momentum"], inplace=True)
    print("Applying actions.")
    df = _apply_actions(df.copy(), num_stocks)
    df = df.dropna(subset=["action"]).reset_index(drop=True)
    print("Weighting actions.")
    df["weight"] = _apply_weights(df.copy())
    print("Completed Strategy Execution.")
    return df
