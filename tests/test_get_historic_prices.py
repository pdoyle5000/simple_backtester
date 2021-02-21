import unittest
import pandas as pd
from momentum_strategy.download_historic_prices import filter_prices


class TestDownloadData(unittest.TestCase):
    def setUp(self):
        self.listing = pd.DataFrame(
            [  # first two entires happen when a merger happens.
                #  if added and removed on the same day, ignore the removed.
                {
                    "dateAdded": "March 22, 2018",
                    "addedSecurity": "IDEXX Laboratories Inc",
                    "removedTicker": "",
                    "removedSecurity": "",
                    "date": "2018-03-22",
                    "reason": "Market capitalization change.",
                    "symbol": "IDXX",
                },
                {
                    "dateAdded": "March 22, 2018",
                    "addedSecurity": "",
                    "removedTicker": "IDXX",
                    "removedSecurity": "IDEXX Laboratories INC",
                    "date": "2018-03-22",
                    "reason": "Market capitalization change.",
                    "symbol": "IDXX",
                },
                {
                    "dateAdded": "March 20, 2017",
                    "addedSecurity": "IDEXX Laboratories Inc",
                    "removedTicker": "",
                    "removedSecurity": "",
                    "date": "2017-03-20",
                    "reason": "Market capitalization change.",
                    "symbol": "IDXX",
                },
                {
                    "dateAdded": "January 7, 1998",
                    "addedSecurity": "",
                    "removedTicker": "IDXX",
                    "removedSecurity": "IDEXX Laboratories Inc",
                    "date": "1998-01-07",
                    "reason": "Market capitalization change",
                    "symbol": "IDXX",
                },
                {
                    "dateAdded": "December 2, 1997",
                    "addedSecurity": "IDEXX Laboratories Inc",
                    "removedTicker": "",
                    "removedSecurity": "",
                    "date": "1997-12-02",
                    "reason": "Market capitalization change.",
                    "symbol": "IDXX",
                },
                {
                    "dateAdded": "December 2, 1997",
                    "addedSecurity": "IDEXX Laboratories Inc",
                    "removedTicker": "",
                    "removedSecurity": "",
                    "date": "1997-12-02",
                    "reason": "Market capitalization change.",
                    "symbol": "TSLA",
                },
                {
                    "dateAdded": "December 21, 2015",
                    "addedSecurity": "",
                    "removedTicker": "GRMN",
                    "removedSecurity": "Garmin Ltd",
                    "date": "2015-12-21",
                    "reason": "Market capitalization change.",
                    "symbol": "GRMN",
                },
            ]
        )

    def test_filter_out_of_index_dates(self):
        mock_prices = pd.DataFrame(
            [
                {"date": pd.to_datetime("1997-12-01"), "symbol": "IDXX", "close": 0},
                {"date": pd.to_datetime("1997-12-02"), "symbol": "IDXX", "close": 1},
                {"date": pd.to_datetime("1998-01-01"), "symbol": "IDXX", "close": 2},
                {"date": pd.to_datetime("1998-01-07"), "symbol": "IDXX", "close": 3},
                {"date": pd.to_datetime("1998-01-08"), "symbol": "IDXX", "close": 0},
                {"date": pd.to_datetime("2000-02-01"), "symbol": "IDXX", "close": 0},
                {"date": pd.to_datetime("2017-03-20"), "symbol": "IDXX", "close": 4},
                {"date": pd.to_datetime("2017-03-21"), "symbol": "IDXX", "close": 5},
                {"date": pd.to_datetime("2020-03-21"), "symbol": "IDXX", "close": 11},
                {"date": pd.to_datetime("2021-03-30"), "symbol": "TSLA", "close": 6},
                {"date": pd.to_datetime("2015-12-20"), "symbol": "GRMN", "close": 0},
                {"date": pd.to_datetime("2015-12-22"), "symbol": "GRMN", "close": 10},
            ]
        )
        expected_prices = pd.DataFrame(
            [
                {"date": pd.to_datetime("2015-12-20"), "symbol": "GRMN", "close": 0},
                {"date": pd.to_datetime("1997-12-02"), "symbol": "IDXX", "close": 1},
                {"date": pd.to_datetime("1998-01-01"), "symbol": "IDXX", "close": 2},
                {"date": pd.to_datetime("1998-01-07"), "symbol": "IDXX", "close": 3},
                {"date": pd.to_datetime("2017-03-20"), "symbol": "IDXX", "close": 4},
                {"date": pd.to_datetime("2017-03-21"), "symbol": "IDXX", "close": 5},
                {"date": pd.to_datetime("2020-03-21"), "symbol": "IDXX", "close": 11},
                {"date": pd.to_datetime("2021-03-30"), "symbol": "TSLA", "close": 6},
            ]
        )
        filtered_prices = filter_prices(self.listing, mock_prices)
        print(filtered_prices)
        # for i, row in self.listing.iterrows():
        #    print(row)
        pd.testing.assert_frame_equal(filtered_prices, expected_prices, check_like=True)
