import unittest
import pandas as pd
from simple_backtester.metrics import (
    cumulative_return,
    annual_return,
    annual_volitility,
    sharpe_ratio,
    max_drawdown,
    stability,
)


class TestMetrics(unittest.TestCase):
    def test_cumulative_returns(self):
        df = pd.DataFrame({"total": [10, 13, 8, 15, 20]})
        target_returns = pd.Series(
            [0.3, -0.2, 0.5, 1.0], index=[1, 2, 3, 4], name="total"
        )
        pd.testing.assert_series_equal(cumulative_return(df), target_returns)

    def test_annual_return(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2018-02-02", "2019-03-03", "2020-05-07", "2020-12-20"]
                ),
                "total": [100, 400, 200, 300],
            }
        )
        self.assertEqual(annual_return(df), 2.0)

    def test_annual_volitility(self):
        df = pd.DataFrame({"total": [100, 125, 150, 175, 200]})
        self.assertEqual(annual_volitility(df), 0.737)

    def test_sharpe_ratio(self):
        df = pd.DataFrame({"total": [100, 98, 100, 104, 105]})
        self.assertEqual(sharpe_ratio(df), 1.197)

    def test_max_drawdown(self):
        df = pd.DataFrame(
            {"total": [10, 15, 20, 25, 23, 22, 30, 35, 20, 15, 7, 25, 26, 27]}
        )
        max_draw_down, _ = max_drawdown(df)
        self.assertEqual(max_draw_down, -0.8)

    def test_stability(self):
        df = pd.DataFrame({"total": [1, 2, 3, 4, 5, 20]})
        self.assertEqual(stability(df), 7.1)
