# Global Package Imports
import numpy as np

def calculate_returns(prices: list) -> list:
    """
    * calculate_returns()
    *
    * Calculates the returns from a list of prices
    *
    * prices: list of asset prices
    * :returns: a list of period returns
    """

    prices = np.array(prices)
    shifted_prices = np.roll(prices, 1)
    shifted_prices[0] = np.nan

    returns = (prices / shifted_prices) - 1

    return returns[1:].tolist()

def calculate_log_returns(prices: list) -> list:
    """
    * calculate_log_returns()
    *
    * Calculates the log returns from a list of prices
    *
    * prices: list of asset prices
    * :returns: a list of period log returns
    """

    prices = np.array(prices)
    shifted_prices = np.roll(prices, 1)

    log_returns = np.log(prices / shifted_prices)

    return log_returns[1:].tolist()

def calculate_asset_returns(portfolio_details: dict):
    """
    * calculate_asset_returns()
    *
    * Calculates the individual asset returns from a complete portfolio
    * The parameter is passed by reference
    *
    * portfolio_details: details of the complete portfolio
    """

    returns = []
    log_returns = []
    asset_prices = portfolio_details['Prices']

    for prices in asset_prices:
        returns.append(calculate_returns(prices))
        log_returns.append(calculate_log_returns(prices))

    # Set the asset returns
    portfolio_details['Returns'] = returns
    portfolio_details['Log_Returns'] = log_returns

def calculate_asset_weights(portfolio_details: dict):
    """
    * calculate_asset_weights()
    *
    * Calculates the individual asset weights from per-share holdings.
    * The parameter is passed by reference
    *
    * portfolio_details: details of the complete portfolio
    """

    weights = []
    total_shares = sum(portfolio_details['Shares'])
    shares = portfolio_details['Shares']

    for i in range(len(shares)):
        # Calculate the weight of each asset
        w = shares[i] / total_shares

        weights.append(w)

    # Set the asset weights
    portfolio_details['Weights'] = weights

def calculate_returns_from_holdings(portfolio_holdings: dict) -> list:
    """
    * extract_returns_from_holdings()
    *
    * Extracts the weighted portfolio return from the portfolio_holdings.
    *
    * portfolio_holdings: dict containing portfolio weights and prices
    * :returns: List of historical portfolio returns, None if error
    """

    portfolio_returns = []

    weights = portfolio_holdings['Weights']
    returns = portfolio_holdings['Returns']

    for i in range(len(returns)):
        r = np.array(returns[i])
        r = r * weights[i]

        if len(portfolio_returns) == 0:
            portfolio_returns = r
        else:
            # Add the weighted portfolio returns
            portfolio_returns = [r1 + r2 for r1, r2 in zip(portfolio_returns, r)]

    return portfolio_returns

def calculate_portfolio_value(portfolio_shares: list, portfolio_prices: list) -> float:
    """
    * calculate_portfolio_value()
    *
    * Calculates the portfolio value from the portfolio shares and prices
    *
    * portfolio_shares: list of portfolio shares
    * portfolio_prices: list of portfolio prices
    * :returns: the value of the portfolio, None if error
    """

    # Extract the most recent prices for each asset
    recent_portfolio_prices = []
    for prices in portfolio_prices:
        recent_portfolio_prices.append(prices[-1])

    # Calculate the portfolio value
    portfolio_value = np.dot(portfolio_shares, recent_portfolio_prices)

    return portfolio_value