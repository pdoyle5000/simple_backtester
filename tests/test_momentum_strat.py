import unittest
import pandas as pd
from simple_backtester.backtester import Action

from momentum_strategy.momentum_strategy import (
    _momentum_score,
    _rolling_groupby,
    _apply_actions,
    _apply_weights,
)

import numpy as np


class TestMomentumStrategy(unittest.TestCase):
    def test_momentum_score(self):
        input_prices = np.array(
            [
                3.0,
                5.0,
                4.0,
                6.0,
                6.5,
                7.0,
                7.5,
                8.8,
                6.9,
                6.7,
                7.0,
                7.1,
                7.4,
                5.2,
                5.1,
                5.0,
                4.8,
                5.1,
                5.1,
            ]
        )
        self.assertEqual(_momentum_score(input_prices), 1.291)
        self.assertEqual(_momentum_score(np.array([0])), -10)

    def test_execute_rolling_groupby(self):
        df = pd.DataFrame(
            [
                {"symbol": "A", "close": 2},
                {"symbol": "A", "close": 3},
                {"symbol": "A", "close": 4},
                {"symbol": "B", "close": 5},
                {"symbol": "B", "close": 6},
                {"symbol": "B", "close": 7},
            ]
        )
        df["means"] = _rolling_groupby(df, np.mean, 3)
        pd.testing.assert_series_equal(
            df["means"],
            pd.Series([np.nan, np.nan, 3.0, np.nan, np.nan, 6.0], name="means"),
        )

    def test_apply_actions(self):
        df = pd.DataFrame(
            [  # Test Buys: Buy A and B and weight properly
                {
                    "symbol": "A",
                    "inv_volatility": 2,
                    "momentum": 1.2,
                    "date": pd.to_datetime("2020-05-07"),
                    "close": 1.0,
                },
                {
                    "symbol": "B",
                    "inv_volatility": 3,
                    "momentum": 1.4,
                    "date": pd.to_datetime("2020-05-07"),
                    "close": 1.0,
                },
                {
                    "symbol": "C",
                    "inv_volatility": 1,
                    "momentum": 1.0,
                    "date": pd.to_datetime("2020-05-07"),
                    "close": 1.0,
                },  # Day 2: Sell A, Buy C, Rebalance B.
                {
                    "symbol": "A",
                    "inv_volatility": 0.5,
                    "momentum": 3.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-10"),
                },
                {
                    "symbol": "B",
                    "inv_volatility": 0.2,
                    "momentum": 4.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-10"),
                },
                {
                    "symbol": "C",
                    "inv_volatility": 0.8,
                    "momentum": 5.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-10"),
                },
            ]
        )

        action_df = pd.DataFrame(
            [
                {
                    "symbol": "A",
                    "date": pd.to_datetime("2020-05-07"),
                    "action": Action.buy,
                    "weight": 0.40,
                    "close": 1.0,
                },
                {
                    "symbol": "B",
                    "date": pd.to_datetime("2020-05-07"),
                    "action": Action.buy,
                    "weight": 0.60,
                    "close": 1.0,
                },
                {
                    "symbol": "A",
                    "date": pd.to_datetime("2020-05-10"),
                    "action": Action.sell,
                    "weight": np.nan,
                    "close": 1.0,
                },
                {
                    "symbol": "B",
                    "date": pd.to_datetime("2020-05-10"),
                    "action": Action.hold,
                    "weight": 0.20,
                    "close": 1.0,
                },
                {
                    "symbol": "C",
                    "date": pd.to_datetime("2020-05-10"),
                    "action": Action.buy,
                    "close": 1.0,
                    "weight": 0.80,
                },
            ]
        )
        actual_df = _apply_actions(df, 2)
        actual_df["weight"] = _apply_weights(actual_df)
        actual_df = (
            actual_df.drop(columns=["momentum", "inv_volatility"])
            .sort_values(by=["date", "symbol"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(actual_df, action_df, check_like=True)

    def test_apply_weights(self):
        test_date = pd.to_datetime("2020-05-07")
        df = pd.DataFrame(
            [  # Test Buys: Buy A and B and weight properly
                {"inv_volatility": 2, "date": test_date, "action": Action.hold},
                {"inv_volatility": 3, "date": test_date, "action": Action.hold},
                {"inv_volatility": 4, "date": test_date, "action": Action.hold},
                {"inv_volatility": 5, "date": test_date, "action": Action.hold},
                {"inv_volatility": 6, "date": test_date, "action": Action.hold},
                {"inv_volatility": 7, "date": test_date, "action": Action.hold},
            ]
        )
        weights = _apply_weights(df)
        self.assertEqual(round(weights.sum(), 1), 1.0)

    def test_drawdown(self):
        df = pd.DataFrame(
            [  # Test Buys: Buy A and B and weight properly
                {
                    "symbol": "A",
                    "inv_volatility": 2,
                    "momentum": 1.2,
                    "date": pd.to_datetime("2020-05-07"),
                    "close": 1.0,
                },
                {
                    "symbol": "B",
                    "inv_volatility": 3,
                    "momentum": 1.4,
                    "date": pd.to_datetime("2020-05-07"),
                    "close": 1.0,
                },
                {
                    "symbol": "C",
                    "inv_volatility": 1,
                    "momentum": 1.0,
                    "date": pd.to_datetime("2020-05-07"),
                    "close": 1.0,
                },  # Day 2: Sell All.
                {
                    "symbol": "A",
                    "inv_volatility": 0.5,
                    "momentum": -100.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-10"),
                },
                {
                    "symbol": "B",
                    "inv_volatility": 0.2,
                    "momentum": -100.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-10"),
                },
                {
                    "symbol": "C",
                    "inv_volatility": 0.8,
                    "momentum": -100.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-10"),
                },  # Day 3: Remain sold off.
                {
                    "symbol": "A",
                    "inv_volatility": 0.5,
                    "momentum": -100.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-11"),
                },
                {
                    "symbol": "B",
                    "inv_volatility": 0.2,
                    "momentum": -100.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-11"),
                },
                {
                    "symbol": "C",
                    "inv_volatility": 0.8,
                    "momentum": -100.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-11"),
                },  # Day 4: reload.
                {
                    "symbol": "A",
                    "inv_volatility": 0.5,
                    "momentum": 100.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-12"),
                },
                {
                    "symbol": "B",
                    "inv_volatility": 0.2,
                    "momentum": 120.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-12"),
                },
                {
                    "symbol": "C",
                    "inv_volatility": 0.8,
                    "momentum": 10.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-12"),
                },
            ]
        )
        pd.testing.assert_series_equal(
            _apply_actions(df, 2).action,
            pd.Series(
                [
                    Action.buy,
                    Action.buy,
                    Action.sell,
                    Action.sell,
                    np.nan,
                    np.nan,
                    Action.buy,
                    Action.buy,
                ]
            ),
            check_names=False,
        )

        # test that no buys happen on the first day if its a down day.
        pd.testing.assert_series_equal(
            _apply_actions(df[3:], 2).action,
            pd.Series([np.nan, np.nan, np.nan, np.nan, Action.buy, Action.buy]),
            check_names=False,
        )

    def test_sell_on_out_of_index(self):
        df = pd.DataFrame(
            [  # Test Buys: Buy A and B and weight properly
                {
                    "symbol": "A",
                    "inv_volatility": 2,
                    "momentum": 1.2,
                    "date": pd.to_datetime("2020-05-07"),
                    "close": 1.0,
                },
                {
                    "symbol": "B",
                    "inv_volatility": 3,
                    "momentum": 1.4,
                    "date": pd.to_datetime("2020-05-07"),
                    "close": 1.0,
                },
                {
                    "symbol": "C",
                    "inv_volatility": 1,
                    "momentum": 1.0,
                    "date": pd.to_datetime("2020-05-07"),
                    "close": 1.0,
                },  # Day 2: Sell A, Buy C, Rebalance B.
                {
                    "symbol": "A",
                    "inv_volatility": 0.5,
                    "momentum": 3.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-10"),
                },
                {
                    "symbol": "B",
                    "inv_volatility": 0.2,
                    "momentum": 4.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-10"),
                },
                {
                    "symbol": "C",
                    "inv_volatility": 0.8,
                    "momentum": 5.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-10"),
                },  # Day 3: Buy A, Hold B, Sell C.
                {
                    "symbol": "A",
                    "inv_volatility": 0.5,
                    "momentum": 3.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-11"),
                },
                {
                    "symbol": "B",
                    "inv_volatility": 0.2,
                    "momentum": 4.0,
                    "close": 1.0,
                    "date": pd.to_datetime("2020-05-11"),
                },
            ]
        )
        actions = _apply_actions(df, 2)
        actions.sort_values(by=["date", "symbol"], inplace=True)
        actions.reset_index(drop=True, inplace=True)
        expected = pd.DataFrame(
            [
                {"symbol": "A", "action": Action.buy},
                {"symbol": "B", "action": Action.buy},
                {"symbol": "A", "action": Action.sell},
                {"symbol": "B", "action": Action.hold},
                {"symbol": "C", "action": Action.buy},
                {"symbol": "A", "action": Action.buy},
                {"symbol": "B", "action": Action.hold},
                {"symbol": "C", "action": Action.sell},
            ]
        )
        pd.testing.assert_frame_equal(
            actions[["symbol", "action"]], expected, check_like=True
        )
