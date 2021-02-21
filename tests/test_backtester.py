import unittest
import numpy as np
import pandas as pd
from copy import deepcopy
from simple_backtester.backtester import BackTester, Action, _rebalance_position


mock_strat = pd.DataFrame(
    [
        {
            "symbol": "A",
            "date": pd.to_datetime("2020-05-07"),
            "action": Action.buy,
            "weight": 0.60,
            "close": 10.00,
        },
        {
            "symbol": "B",
            "date": pd.to_datetime("2020-05-07"),
            "action": Action.buy,
            "weight": 0.40,
            "close": 30.00,
        },
        {
            "symbol": "A",
            "date": pd.to_datetime("2020-05-10"),
            "action": Action.sell,
            "weight": np.nan,
            "close": 25.00,
        },
        {
            "symbol": "B",
            "date": pd.to_datetime("2020-05-10"),
            "action": Action.hold,
            "weight": 0.80,
            "close": 50.00,
        },
        {
            "symbol": "C",
            "date": pd.to_datetime("2020-05-10"),
            "action": Action.buy,
            "weight": 0.20,
            "close": 100.00,
        },
    ]
)


class TestBacktester(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.strat = mock_strat.copy()
        self.test_date = pd.to_datetime("2020-05-10")
        self.df = self.strat.query("date == @self.test_date").reset_index(drop=True)
        self.yesterdays_summary = {
            self.test_date: {
                "cash": 10.0,
                "investments": 990.0,
                "total": 1000.0,
                "shares_owned": {
                    "A": {"num_shares": 60, "close": 10.00},
                    "B": {"num_shares": 13, "close": 30.00},
                },
            }
        }

    def test_backtester(self):
        backtester = BackTester(self.strat, 1000.00)
        expected_daily_state = {
            pd.to_datetime("2020-05-07"): {
                "cash": 10.0,
                "investments": 990.0,
                "total": 1000.0,
                "shares_owned": {
                    "A": {"num_shares": 60.0, "close": 10.00},
                    "B": {"num_shares": 13.0, "close": 30.00},
                },
            },
            pd.to_datetime("2020-05-10"): {
                "cash": 60.0,
                "investments": 2100.0,
                "total": 2160.0,
                "shares_owned": {
                    "B": {"num_shares": 34.0, "close": 50.00},
                    "C": {"num_shares": 4.0, "close": 100.00},
                },
            },
        }
        self.assertDictEqual(backtester.daily_state, expected_daily_state)

    def test_rebalancer_buying(self):
        to_rebalance = deepcopy(self.yesterdays_summary)
        rebalanced = {
            self.test_date: {
                "cash": -1040.0,
                "investments": 2300.0,
                "total": 1260.0,
                "shares_owned": {
                    "A": {"num_shares": 60, "close": 10.00},
                    "B": {"num_shares": 34, "close": 50.00},
                },
            }
        }
        _rebalance_position(self.df, 1, to_rebalance)
        self.assertDictEqual(to_rebalance, rebalanced)
        self.assertEqual(self.df.at[1, "value"], 50 * 34)
        self.assertEqual(self.df.at[1, "num_shares"], 34)

    def test_rebalancer_selling(self):
        to_rebalance = deepcopy(self.yesterdays_summary)
        self.df.at[1, "weight"] = 0.20
        self.df.at[2, "weight"] = 0.80
        rebalanced = {
            self.test_date: {
                "cash": 260.0,
                "investments": 1000.0,
                "total": 1260.0,
                "shares_owned": {
                    "A": {"num_shares": 60, "close": 10.00},
                    "B": {"num_shares": 8, "close": 50.00},
                },
            }
        }
        _rebalance_position(self.df, 1, to_rebalance)
        self.assertDictEqual(to_rebalance, rebalanced)
        self.assertEqual(self.df.at[1, "value"], 8 * 50)
        self.assertEqual(self.df.at[1, "num_shares"], 8)

    def test_rebalancer_noop(self):
        to_rebalance = deepcopy(self.yesterdays_summary)
        self.df.at[1, "weight"] = 0.32
        self.df.at[2, "weight"] = 0.68
        rebalanced = {
            self.test_date: {
                "cash": 10.0,
                "investments": 1250.0,
                "total": 1260.0,
                "shares_owned": {
                    "A": {"num_shares": 60, "close": 10.00},
                    "B": {"num_shares": 13, "close": 50.00},
                },
            }
        }
        _rebalance_position(self.df, 1, to_rebalance)
        self.assertDictEqual(to_rebalance, rebalanced)
        self.assertEqual(self.df.at[1, "value"], 13 * 50)
        self.assertEqual(self.df.at[1, "num_shares"], 13)

    @unittest.skip("")
    def test_backtester_metric_generation(self):
        backtester = BackTester(self.simple_strategy, 1000.00)
        target_metrics = {"annual_return": -0.11}
        self.assertDictEqual(backtester.metrics, target_metrics)

    @unittest.skip("")
    def test_summarize_investments(self):
        backtester = BackTester(self.strat, 1000.00)
        # summary = summarize_investments(backtester.daily_state)
        summary = None
        expected_summary = pd.DataFrame(
            [
                {
                    "buy_date": pd.to_datetime("2020-01-01"),
                    "sell_date": pd.to_datetime("2020-01-02"),
                    "days_held": 0,
                    "symbol": "A",
                    "purchase_price": 5,
                    "sell_price": 9,
                    "raw_return": 4,
                    "tax": 1,
                    "net_return": 3,
                    "fees": 0,
                }
            ]
        )
        pd.testing.assert_frame_equal(summary, expected_summary, check_like=True)
