# Global Package Imports
import math
import numpy as np
from sklearn.linear_model import LinearRegression

# Local Package Imports
from RiskEngine._RiskEngine import RiskEngine
from RiskEngine._utils import (
    calculate_portfolio_value
)

class StockRiskEngine(RiskEngine):
    """
    * StockRiskEngine(RiskEngine)
    *
    * The stock risk engine is the framework for calculating all financial risk
    * metrics associated with an equity-based portfolio.
    """

    # Constants
    _TRADING_DAYS = 252

    def __init__(self, portfolio_details: dict, market_prices: list):
        """
        * __init__()
        *
        * Initializes the stock risk engine and the base class.
        *
        * portfolio_details: details of the financial portfolio (symbols, shares, prices)
        *                    each dict value index is matched accross all keys
        *   NOTE: portfolio_details dict must be in the form:
        *         {
        *           "Symbols" : [Symbol_1, Symbol_2, ..., Symbol_N],
        *           "Shares" : [Shares_1, Shares_2, ..., Shares_N],
        *           "Prices" : [[Prices_1], [Prices_2], ..., [Prices_N]]
        *         }
        * market_prices: historical prices of the market portfolio (benchmark)
        *   NOTE: the market returns must be at least as long as the portfolio returns
        """

        RiskEngine.__init__(self, portfolio_details, market_prices)

    def beta(self, portfolio_returns: list, market_returns: list) -> float:
        """
        * beta()
        *
        * Linear, first-order risk metric for stocks.
        * Beta measures the portfolio's return sensitivity to the benchmark.
        * Calculated by finding the slope of the regression between the portfolio
        * and market returns.
        * NOTE: The size of the market returns MUST be the same periocidy of the
        *       portfolio and have at least as many occurances as the portfolio.
        *
        * market_returns: list of market returns
        * :returns: beta of the portfolio, None if failure
        """

        beta = None
        N = len(portfolio_returns)

        if len(market_returns) < N:
            raise ValueError("Insufficient market returns...")

        else:
            # Format the returns
            market_returns = market_returns[:N]
            X = np.array(portfolio_returns).reshape(-1, 1)
            y = np.array(market_returns)

            linear_model = LinearRegression()
            linear_model.fit(X, y)

            beta = linear_model.coef_[0] # Extract beta coefficient

        return beta

    def IndividualVAR(self, symbol: str, confidence_interval: float = 0.99, log_based: bool = False) -> float:
        """
        * IndividualVAR()
        *
        * Calculates the individual value at risk (VAR) for an asset using the historical
        * asset returns.
        * The individual VAR is calculated as:
        *  VAR = sd * z * w
        *
        * confidence_interval: confidence interval for the VAR calculation
        * log_based: if True, use log returns, else use simple returns
        * :returns: the individual VAR, None if failure
        """

        # Validate confidence interval
        if confidence_interval <= 0.01 or confidence_interval >= 1:
            raise ValueError('Confidence interval must be greater than 0 and less than 1...')
            return None

        # Get the asset returns
        if log_based:
            asset_return = self.get_asset_log_returns(symbol)
        else:
            asset_return = self.get_asset_returns(symbol)

        # Get the asset weight
        asset_weight = self.get_asset_weight(symbol)

        # Calculate the annualized individual portfolio risk
        stddev = self.standard_deviation(asset_return) * math.sqrt(self._TRADING_DAYS)

        if dollar_based:
            # Get the asset price and shares
            recent_price = self.get_asset_prices(symbol)[-1]
            tota_shares = self.get_asset_shares(symbol)

            # Calculate the individual VAR
            var = stddev * self.critical_z_score(confidence_interval)

            # Return the dollar-based VAR
            return var * recent_price * total_shares
        else:
            # Calculate the individual VAR
            var = stddev * self.critical_z_score(confidence_interval) * asset_weight

            return var

    def BasicPortfolioVAR(self, confidence_interval: float = 0.99, log_based: bool = False, dollar_based: bool = False) -> float:
        """
        * BasicPortfolioVAR()
        *
        * Calculates the basic portfolio value at risk (VAR) using the covariance
        * matrix of the portfolio's assets. This method calculates VAR based on its
        * historical asset returns.
        * The portfolio variance is calclated as:
        *  V = w * COV * w'
        * The portfolio risk is then calculated as:
        *  R = sqrt(V)
        * The VAR is then calculated as:
        *  VAR = R * z
        * where z is the z-score of the confidence interval.
        *
        * confidence_interval: confidence interval for the VAR calculation
        * log_based: if True, use log returns, else use simple returns
        * dollar_based: if True, use dollar based VAR, else use percentage based VAR
        * :returns: the basic portfolio VAR, None if failure
        """

        # Validate confidence interval
        if confidence_interval <= 0.01 or confidence_interval >= 1:
            raise ValueError('Confidence interval must be greater than 0 and less than 1...')
            return None

        # Calculate the coariance matrix
        if log_based:
            cov_matrix = np.cov(self.portfolio_asset_log_returns, rowvar = True)
        else:
            cov_matrix = np.cov(self.portfolio_asset_returns, rowvar = True)

        # Variance of portfolio rate of return
        portfolio_variance = self.portfolio_weights @ cov_matrix @ np.transpose(self.portfolio_weights)

        # Callculate the portfolio risk
        portfolio_risk = math.sqrt(portfolio_variance)

        # Calculate and return the basic VAR
        var = portfolio_risk * self.critical_z_score(confidence_interval)

        if dollar_based:
            return var * calculate_portfolio_value(self.portfolio_shares, self.portfolio_prices)
        else:
            return var

    def ConditionalLocalVAR(self):
        pass

    def MarginalLocalVAR(self):
        pass