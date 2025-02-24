
def validate_portfolio_holdings(portfolio_holdings: dict) -> bool:
    # TODO: Validate that weights are floats, each float has a list
    return True

def portfolio_returns_from_holdings(portfolio_holdings: dict) -> list:
    """
    * extract_returns_from_holdings()
    *
    * Extracts the weighted portfolio return from the portfolio_holdings.
    *
    * portfolio_holdings: dict containing portfolio weights and returns
    * :returns: List of historical portfolio returns, None if error
    """

    portfolio_weights = list(portfolio_holdings.keys())
    position_returns = list(portfolio_holdings.values())

    portfolio_returns = []

    # Calclated the weighted returns
    for i in range(len(portfolio_weights)):
        weighted_return = [ret * portfolio_weights[i] for ret in position_return[i]]

        if len(portfolio_returns) != 0:
            portfolio_returns = [x + y for x, y in zip(portfolio_returns, weighted_return)]

        else:
            portfolio_returns = weighted_return

    return portfolio_returns