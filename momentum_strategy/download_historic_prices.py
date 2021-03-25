import pandas as pd
import sys
import os
from time import sleep
from tqdm.contrib.concurrent import thread_map
from typing import Optional
from functools import partial
from datetime import date
from urllib.error import HTTPError

BASEURL = "https://financialmodelingprep.com/api/v3/"


# Future Improvement: Right now we get the nasdaq listing
# then we get full price histories and then use the listing
# to filter on the times when that stock was in the nasdaq.
# we could use the nasdaq times to just get the right time slices from
# the API. (we could just not filter and keep the companies too).


def get_index_listing(
    api_key: str, base_url: str, index: str, historical: bool = False
) -> pd.DataFrame:
    # query api
    if historical:  # make the false option download just today.
        base_url = base_url + "historical/"
    endpoint = base_url + f"{index}_constituent?apikey={api_key}"
    df = pd.read_json(endpoint)
    return df


def _get_raw_prices(symbol: str, api_key: str) -> pd.DataFrame:
    today = str(date.today())
    endpoint = f"{BASEURL}/historical-price-full/{symbol}?from=1997-12-02&to={today}&apikey={api_key}&datatype=csv"  # noqa: E501
    result = None
    while result is None:
        try:
            df = pd.read_csv(endpoint)
            df["symbol"] = symbol
            df["date"] = pd.to_datetime(df.date)
            if len(df) > 5:
                return df
        except HTTPError:
            sleep(0.5)
            continue
        result = 1


def get_daily_prices(
    api_key: str, base_url: str, historical: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    if historical is not None:
        price_func = partial(_get_raw_prices, api_key=api_key)
        df = pd.concat(
            thread_map(
                price_func, historical.symbol.unique(), desc="get symbol prices."
            )
        )
    return df


def filter_prices(listing: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Filters price df based on timeframes of existence within the listing df.

    The listing index on financialmodelingprep.com lists dates that companies
    leave by populating removedSecurity and are added with addedSecurity.
    This function will sort that index and every time an add is found, wait for
    the removed (or current date) to build a query string for that symbol.

    There is some added complexity where a merger may happen ex: (ATVI or VMED).
    This will show up as the company being removed and added on the same day if
    it was already previously in the index.

    This function passes the unit tests but is way past need for the TDD refactor
    iteration.

    """
    filtered_listing = pd.DataFrame()

    listing["date"] = pd.to_datetime(listing.date)
    listing.sort_values(by=["date", "removedTicker"], inplace=True)

    # generate list of query strings.
    queries = []
    for symbol, company_df in listing.groupby("symbol"):
        queries.extend(_get_indexed_time_ranges(symbol, company_df))
    for query_dict in queries:
        start = query_dict["start"]  # noqa: F841
        if start is None:
            start = pd.to_datetime("1997-12-02")
        end = query_dict["end"]
        symbol = query_dict["symbol"]
        if end:  # has an end date.
            filtered_listing = filtered_listing.append(
                prices.query("symbol == @symbol and date >= @start and date <= @end")
            )
        else:  # no end date, one sided date query.
            filtered_listing = filtered_listing.append(
                prices.query("symbol == @symbol and date >= @start")
            )
    return filtered_listing.reset_index(drop=True)


def _get_indexed_time_ranges(symbol: str, df: pd.DataFrame) -> list:
    queries = []
    end_date = None
    start_date = None
    double_start_date = None
    for i, row in df.iterrows():
        if row.addedSecurity:
            if start_date is not None:  # in a MnA mode.
                double_start_date = row.date
            else:
                start_date = row.date
        if row.removedTicker == symbol:
            end_date = row.date
            if end_date == double_start_date:
                end_date = None
                continue
            queries.append({"start": start_date, "end": end_date, "symbol": symbol})
            start_date = None
            end_date = None
    if start_date and not end_date:
        queries.append({"start": start_date, "end": None, "symbol": symbol})
    return queries


if __name__ == "__main__":
    api_key = sys.argv[1]
    print(f"Using API key {api_key}")
    os.makedirs("../data/", exist_ok=True)

    qqqspy = pd.DataFrame({"symbol": ["QQQ", "SPY"]})
    get_daily_prices(api_key, BASEURL, historical=qqqspy).to_csv(
        f"QQQ_SPY.csv", index=False
    )
    for index in ["dowjones", "nasdaq", "sp500"]:
        print(f"Downloading {index.upper()}")
        listing = get_index_listing(api_key, BASEURL, index=index, historical=True)
        print("getting prices...")
        prices = get_daily_prices(api_key, BASEURL, historical=listing)
        prices.to_csv(f"../data/{index}_prices.csv", index=False)
        print("filtering prices...")
        filtered_prices = filter_prices(listing, prices)
        print("Complete outputting csv.")
        filtered_prices.to_csv(f"../data/{index}_backtest.csv", index=False)
