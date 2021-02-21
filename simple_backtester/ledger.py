from enum import Enum, unique


@unique
class Transaction(Enum):
    buy = "buy"
    sell = "sell"
    rebalance_buy = "rebalance_buy"
    rebalance_sell = "rebalance_sell"


# add transaction  TODO: Create Detailed Ledger functions to
# track rebalacing and real tax/loss/gain


#    purchase_total = _get_purchase_total(transactions)
#    nbt = revenue - purchase_total
#    transactions.loc[len(transactions)] = {
#        "transaction_type": Transaction.sell,
#        "date": date,
#        "symbol": symbol,
#        "num_shares": num_shares,
#        "price": close,
#        "dollars_transacted": revenue,
#        "purchase_total": purchase_total,
#        "nbt": nbt,
#        "tax": tax_rate * nbt,
#        "fees": fees,
#        "net": nbt * (1 - tax_rate) - fees,
#    }

# def _get_purchase_total(transactions: pd.DataFrame, symbol: str) -> float:
#    # look at all the transactions for this symbol since the latest buy and tally dollars spent.
#    ledger = transactions.query("symbol == @symbol")
#    last_buy = ledger.query("transaction_type == @Transaction.buy").index[-1]
#    purchase_total = 0
#    for i, row in ledger[last_buy:].iterrows():
#        if row.transaction_type in [Transaction.buy, Transaction.rebalance_buy]:
#            purchase_total += row.dollars_transacted
#        if row.transaction_type == Transaction.rebalance_sell:
#            purchase_total -= row.purchase_total
#    return 2

# The backtester needs a ledger class to track complex details about
# profits and losses for stocks taking rebalancing and taxes into account.
# This is the next big thing and may be its own class (at least its own file)
# self.transactions = pd.DataFrame(
#    columns=[
#        "transaction_type",
#        "date",
#        "symbol",
#        "buy_price",
#        "sell_price",
#        "days_held",
#        "gross",
#        "tax",
#        "fees",
#        "net",
#        "is_rebalance",
#    ]
# )
# self.transactions["is_rebalance"] = self.transactions.is_rebalance.astype(bool)
# self.transactions["buy_date"] = pd.to_datetime(self.transactions.buy_date)
# self.transactions["sell_date"] = pd.to_datetime(self.transactions.sell_date)
