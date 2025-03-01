# Global Package Imports
import math
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# Local Package Imports
from RiskEngine._RiskEngine import RiskEngine

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
        * portfolio_details: details of the financial portfolio (symbols, weights, prices)
        *                    each dict value index is matched accross all keys
        *   NOTE: portfolio_details dict must be in the form:
        *         {
        *           "Symbols" : [Symbol_1, Symbol_2, ..., Symbol_N],
        *           "Weights" : [Weight_1, Weight_2, ..., Weight_N],
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

    def BasicLocalVAR(self, confidence_interval: float = 0.99) -> float:

        # Validate confidence interval
        if confidence_interval <= 0 or confidence_interval >= 1:
            raise ValueError('Confidence interval must be greater than 0 and less than 1...')
            return None

        # Get the z-score from the confidence interval
        alpha = 1 - confidence_interval
        z_score = stats.norm.ppf(1 - alpha / 2)

        # Calculate the annualized standard deviation for each asset's return
        stddevs = []
        for returns in self.portfolio_asset_log_returns:
            stddev = self.standard_deviation(returns)
            stddevs.append(stddev * math.sqrt(len(returns) / self._TRADING_DAYS))

        # Calculate the coariance matrix
        cov_matrix = np.cov(self.portfolio_asset_log_returns, rowvar = True)

        adjusted_covariance = np.matmul(cov_matrix, self.portfolio_weights)

        # Calculate individual positional risk
        positional_risk = [adjusted_covariance[i] * self.portfolio_weights[i] for i in range(len(adjusted_covariance))]
        total_positional_risk = sum(positional_risk)

        risk = math.sqrt(total_positional_risk)

        # Calculate and return the basic VAR
        return risk * z_score

    def ConditionalLocalVAR(self):
        pass

    def MarginalLocalVAR(self):
        pass