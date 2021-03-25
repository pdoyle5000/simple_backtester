import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from simple_backtester.backtester import update_ledger, _init_ledger, Action


class TestUpdateLedger(unittest.TestCase):
    def setUp(self):
        self.ledger = _init_ledger()

    def test_simple_buy(self):
        update_ledger(
            self.ledger,
            symbol="abc",
            num_shares=6,
            date=pd.to_datetime("2021-05-07"),
            close=5.5,
        )
        expected_df = pd.DataFrame(
            [
                {
                    "date": pd.to_datetime("2021-05-07"),
                    "symbol": "abc",
                    "action": Action.buy,
                    "num_shares": 6,
                    "close": 5.5,
                    "gain": np.nan,
                    "tax": np.nan,
                    "fees": np.nan,
                    "net_gain": np.nan,
                    "shares_owned": 6,
                    "is_rebalance": False,
                }
            ]
        )
        pd.testing.assert_frame_equal(expected_df, self.ledger, check_like=True)

    def test_simple_sell(self):
        # install a buy.
        update_ledger(
            self.ledger,
            symbol="abc",
            num_shares=6,
            date=pd.to_datetime("2021-05-07"),
            close=5.5,
        )
        update_ledger(
            self.ledger,
            symbol="abc",
            num_shares=-2,
            date=pd.to_datetime("2021-05-07"),
            close=8.0,
        )
        self.assertEqual(self.ledger.loc[1].symbol, "abc")
        self.assertEqual(self.ledger.loc[1].num_shares, 2)
        self.assertEqual(self.ledger.loc[1].action, Action.sell)
        self.assertEqual(self.ledger.loc[1].gain, 5.0)
        self.assertEqual(self.ledger.loc[1].tax, 5.0 * 0.22)
        self.assertEqual(self.ledger.loc[0].shares_owned, 4)
        self.assertLess(self.ledger.loc[1].net_gain, 5.0 - self.ledger.loc[1].tax)

    def test_complex_sell(self):
        # sell order that is for more than the latest buy.
        # install buys.
        update_ledger(
            self.ledger,
            symbol="efg",
            num_shares=6,
            date=pd.to_datetime("2021-05-07"),
            close=5.5,
        )
        update_ledger(
            self.ledger,
            symbol="abc",
            num_shares=6,
            date=pd.to_datetime("2021-05-07"),
            close=5.0,
        )
        update_ledger(
            self.ledger,
            symbol="hij",
            num_shares=6,
            date=pd.to_datetime("2021-05-07"),
            close=5.5,
        )
        update_ledger(
            self.ledger,
            symbol="abc",
            num_shares=4,
            date=pd.to_datetime("2021-05-08"),
            close=2.0,
        )
        update_ledger(
            self.ledger,
            symbol="abc",
            num_shares=2,
            date=pd.to_datetime("2021-05-09"),
            close=3.0,
        )
        update_ledger(
            self.ledger,
            symbol="abc",
            num_shares=-8,
            date=pd.to_datetime("2021-05-09"),
            close=10.0,
        )
        self.assertEqual(self.ledger.loc[5].symbol, "abc")
        self.assertEqual(self.ledger.loc[5].num_shares, 8)
        self.assertEqual(self.ledger.loc[1].shares_owned, 0)
        self.assertEqual(self.ledger.loc[3].shares_owned, 2)
        self.assertEqual(self.ledger.loc[5].action, Action.sell)
        self.assertEqual(self.ledger.loc[5].gain, 46.0)
        self.assertEqual(self.ledger.loc[5].tax, 46.0 * 0.22)
        self.assertLess(self.ledger.loc[5].net_gain, 46.0 - self.ledger.loc[5].tax)
