import pandas as pd
import numpy as np
from typing import Tuple


def percent_return(daily_df: pd.DataFrame) -> float:
    df = daily_df.iloc[[0, -1]]
    return (df.iloc[1].total - df.iloc[0].total) / df.iloc[0].total


def annual_return(daily_df: pd.DataFrame) -> float:
    df = daily_df.iloc[[0, -1]]
    months = max(
        (df.iloc[0]["datetime"] - df.iloc[1]["datetime"]) / np.timedelta64(1, "M"), 1
    )
    return round(((percent_return(daily_df) + 1) ** (1 / months)) - 1, 2)


def cumulative_return(daily_df: pd.DataFrame) -> pd.Series:
    return (1 + daily_df.total.pct_change()[1:]).cumprod() - 1


def annual_volitility(daily_df: pd.DataFrame) -> float:
    trading_days_yr = 252
    return round(daily_df.total.pct_change()[1:].std() * np.sqrt(trading_days_yr), 3)


def sharpe_ratio(daily_df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    # https://www.investopedia.com/terms/s/sharperatio.asp
    # (return - risk free rate) / std of return
    # not bond investing, so risk-free-rate is 2% (inflation)
    portfolio_return = percent_return(daily_df)
    std = daily_df.total.pct_change()[1:].std()
    return round((portfolio_return - risk_free_rate) / std, 3)


def stability(daily_df: pd.DataFrame) -> float:
    return round(daily_df.total.std(), 1)


def max_drawdown(daily_df: pd.DataFrame) -> Tuple[float, pd.Series]:
    wealth_index = 1000 * (1 + daily_df.total.pct_change()[1:]).cumprod()
    drawdown = (wealth_index - wealth_index.cummax()) / wealth_index.cummax()
    return (round(drawdown.min(), 3), drawdown)


# TODOS: write these


def _calmar_ratio(daily_df: pd.DataFrame) -> float:
    return 2


def _omega_ratio(daily_df: pd.DataFrame) -> float:
    return 2


def _sortino_ratio(daily_df: pd.DataFrame) -> float:
    return 2


def _kurtosis(daily_df: pd.DataFrame) -> float:
    return 2


def _skew(daily_df: pd.DataFrame) -> float:
    return 2


def _tail_ratio(daily_df: pd.DataFrame) -> float:
    return 2


def _daily_value_risk(daily_df: pd.DataFrame) -> float:
    return 2
