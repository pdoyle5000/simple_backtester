from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from enum import Enum, unique
from datetime import timedelta, datetime
from copy import deepcopy
from tqdm import tqdm

from simple_backtester.metrics import annual_return


FINRA_FEE = 0.0000051
AGG_SALES_FEE = 0.000119


@unique
class Action(Enum):
    hold = 1
    sell = 2
    buy = 3


def sell(df: pd.DataFrame, i: int, daily_state: dict, ledger: pd.DataFrame) -> None:
    date = df.at[i, "date"]
    symbol = df.at[i, "symbol"]
    num_shares = daily_state[date]["shares_owned"][symbol]["num_shares"]
    close = df.at[i, "close"]

    # sell shares into cash
    revenue = num_shares * close
    daily_state[date]["cash"] += revenue
    df.at[i, "value"] = revenue

    # remove from investments
    daily_state[date]["investments"] -= revenue

    # remove shares from books
    df.at[i, "num_shares"] = num_shares
    daily_state[date]["shares_owned"].pop(symbol)

    # update ledger
    update_ledger(ledger, symbol=symbol, date=date, num_shares=-num_shares, close=close)

    # update total
    daily_state[date]["total"] = (
        daily_state[date]["investments"] + daily_state[date]["cash"]
    )


def buy(df: pd.DataFrame, i: int, daily_state: dict, ledger: pd.DataFrame) -> None:
    # calculate the number of shares to buy
    date = df.at[i, "date"]
    day_start_total = daily_state[date]["total"]
    close = df.at[i, "close"]
    num_shares = day_start_total * df.at[i, "weight"] // close
    symbol = df.at[i, "symbol"]
    df.at[i, "num_shares"] = num_shares
    daily_state[date]["shares_owned"][symbol] = {
        "num_shares": num_shares,
        "close": close,
    }

    # add to value
    value = close * num_shares
    df.at[i, "value"] = value
    daily_state[date]["investments"] += value

    # subtract from cash
    daily_state[date]["cash"] -= value

    # update ledger
    update_ledger(ledger, symbol=symbol, date=date, num_shares=num_shares, close=close)

    # update total
    daily_state[date]["total"] = (
        daily_state[date]["cash"] + daily_state[date]["investments"]
    )


def hold(
    df: pd.DataFrame, i: int, daily_state: dict, rebalance_df: pd.DataFrame
) -> None:
    # instead of rebalancing every day, set a cadence.
    # if no rebalance, just update hold totals.
    _rebalance_position(df, i, daily_state, rebalance_df)


def _rebalance_position(
    df: pd.DataFrame, i: int, daily_state: dict, ledger: pd.DataFrame
) -> None:
    # calculate an updated total for weighting
    date = df.at[i, "date"]
    total = daily_state[date]["cash"]
    for owned_stock in daily_state[date]["shares_owned"].keys():
        owned_stock_df = df.query("symbol == @owned_stock")
        total += float(
            owned_stock_df.close.values
            * daily_state[date]["shares_owned"][owned_stock]["num_shares"]
        )

    todays_close = df.at[i, "close"]
    symbol = df.at[i, "symbol"]
    rebalance_weight = df.at[i, "weight"]

    # calculate appropriate number of shares to rebalance too.
    old_num_shares = daily_state[date]["shares_owned"][symbol]["num_shares"]
    old_investment = float(
        daily_state[date]["shares_owned"][symbol]["close"] * old_num_shares
    )

    new_num_shares = int((total * rebalance_weight) // todays_close)
    new_investment = new_num_shares * todays_close

    share_diff = new_num_shares - old_num_shares
    if share_diff != 0:
        update_ledger(
            ledger,
            symbol=symbol,
            date=date,
            num_shares=share_diff,
            close=todays_close,
            is_rebalance=True,
        )

    # update cash total
    daily_state[date]["cash"] -= (new_num_shares - old_num_shares) * todays_close

    # update todays state.
    daily_state[date]["shares_owned"][symbol]["num_shares"] = new_num_shares
    daily_state[date]["shares_owned"][symbol]["close"] = todays_close
    daily_state[date]["investments"] += new_investment - old_investment
    daily_state[date]["total"] = (
        daily_state[date]["investments"] + daily_state[date]["cash"]
    )
    df.at[i, "value"] = new_num_shares * todays_close
    df.at[i, "num_shares"] = new_num_shares


def update_ledger(
    ledger: pd.DataFrame,
    *,
    symbol: str,
    num_shares: int,
    date: datetime,
    close: float,
    is_rebalance: bool = False,
) -> None:
    gain: float = np.nan
    tax: float = np.nan
    fees: float = np.nan
    action: Action = Action.buy
    if num_shares < 0:
        action = Action.sell
        gain = _calculate_gains(ledger, symbol, abs(num_shares), close)
        tax = gain * 0.22
        fees = (FINRA_FEE * close * abs(num_shares)) + (
            AGG_SALES_FEE * close * abs(num_shares)
        )

    transaction = {
        "date": date,
        "symbol": symbol,
        "action": action,
        "close": close,
        "num_shares": abs(num_shares),
        "gain": gain,
        "tax": tax,
        "fees": fees,
        "net_gain": gain - tax - fees,
        "shares_owned": abs(num_shares),  # Stateful
        "is_rebalance": is_rebalance,
    }
    ledger.loc[len(ledger)] = transaction


def _calculate_gains(
    ledger: pd.DataFrame, symbol: str, num_shares: int, close: float
) -> float:
    buy_list = ledger.query(
        "symbol == @symbol and action == @Action.buy and shares_owned > 0"
    )
    shares_left_to_sell = abs(num_shares)

    if buy_list.shares_owned.sum() < shares_left_to_sell:
        raise ValueError("Trying to sell more shares than you own.")
    gain: float = 0.0
    for i, row in buy_list.iterrows():
        if row.num_shares >= shares_left_to_sell:
            gain += shares_left_to_sell * (close - row.close)
            ledger.at[i, "shares_owned"] -= shares_left_to_sell
            return gain
        shares_owned = ledger.at[i, "shares_owned"]
        gain += shares_owned * (close - row.close)
        shares_left_to_sell -= shares_owned
        ledger.at[i, "shares_owned"] = 0

    # should never get here.
    return gain


# hunt through df for the last time the stock was purchased.
def _init_ledger() -> pd.DataFrame:
    return (
        pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "action",
                "close",
                "num_shares",
                "gain",
                "tax",
                "fees",
                "net_gain",
                "shares_owned",
                "is_rebalance",
            ],
            index=[0],
        )
        .dropna()
        .astype(
            {
                "symbol": "str",
                "close": "float",
                "num_shares": "int",
                "shares_owned": "int",
                "is_rebalance": "bool",
            }
        )
        .reset_index(drop=True)
    )


class BackTester:
    def __init__(
        self,
        strategy: pd.DataFrame,
        bankroll: float,
        start_date: Optional[str] = None,
        buy_col: str = "close",
        sell_col: str = "close",
    ):
        dependent_cols = list(
            set(["symbol", "weight", "action", "date", buy_col, sell_col])
        )
        for col in dependent_cols:
            if col not in strategy:
                print(f"{col} does not exist, cannot execute backtest.")
                raise (KeyError)
        self.bankroll = bankroll
        self.buy_col = buy_col
        self.sell_col = sell_col

        if start_date:
            start = pd.to_datetime(start_date)  # noqa: F841
            self.strategy = strategy.query("date > @start")
        self.ledger = _init_ledger()
        self.data, self.daily_state = self.execute_backtest(strategy)
        self.metrics = self.calculate_metrics(self.daily_state)

    def init_execution_cols(self, strat_df: pd.DataFrame) -> None:
        strat_df["action"] = pd.Categorical(
            strat_df["action"], [a for a in Action], ordered=True
        )
        strat_df["value"] = 0
        strat_df["num_shares"] = 0
        strat_df.sort_values(by=["date", "action"], inplace=True)
        strat_df.reset_index(drop=True, inplace=True)

    def execute_backtest(
        self, strat_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[datetime, dict]]:
        self.init_execution_cols(strat_df)
        action_map = {Action.sell: sell, Action.buy: buy, Action.hold: hold}

        daily_state: Dict[pd.datetime, dict] = {
            strat_df.at[0, "date"]: {
                "cash": self.bankroll,
                "investments": 0,
                "total": self.bankroll,
                "shares_owned": {},
            }
        }

        daily_actions_df = pd.DataFrame()
        for day, df in tqdm(strat_df.groupby("date"), desc="Daily Backtest"):
            _seed_today(
                day=day, strat_df=strat_df, today_df=df, daily_state=daily_state
            )
            for i, row in df.iterrows():
                action_map[row.action](df, i, daily_state, self.ledger)

            daily_actions_df = pd.concat([daily_actions_df, df]).reset_index(drop=True)

        return daily_actions_df, daily_state

    def calculate_metrics(self, daily_state: Dict[datetime, dict]) -> Dict[str, float]:
        df = pd.DataFrame()
        for key, day_dict in daily_state.items():
            day_df = pd.DataFrame([{"datetime": key, "total": day_dict["total"]}])
            df = pd.concat([df, day_df])

        self.daily_totals = df
        return {"annual_return": annual_return(df)}


def _seed_today(
    *, day: datetime, strat_df: pd.DataFrame, today_df, daily_state: dict
) -> None:
    if day != strat_df.at[0, "date"]:
        previous_day = _get_previous_day(day, daily_state)
        daily_state[day] = deepcopy(daily_state[previous_day])
        # update closes for the day.
        todays_investment_total = 0.0
        for i, row in today_df.iterrows():
            if row.action != Action.buy:
                daily_state[day]["shares_owned"][row.symbol]["close"] = row.close
                todays_investment_total += (
                    row.close
                    * daily_state[day]["shares_owned"][row.symbol]["num_shares"]
                )
        daily_state[day]["investments"] = todays_investment_total
        daily_state[day]["total"] = (
            daily_state[day]["cash"] + daily_state[day]["investments"]
        )


def _get_previous_day(day: datetime, daily_state: dict) -> datetime:
    for days_before in range(1, 100):  # some impossible range.
        if day - timedelta(days=days_before) in daily_state:
            return day - timedelta(days=days_before)
    return day
