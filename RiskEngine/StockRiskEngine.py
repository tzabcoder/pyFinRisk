# Global Package Imports
import numpy as np
from sklearn.linear_model import LinearRegression

# Local Package Imports
from _RiskEngine import RiskEngine

class StockRiskEngine(RiskEngine):
    """
    * StockRiskEngine(RiskEngine)
    *
    * The stock risk engine is the framework for calculating all financial risk
    * metrics associated with an equity-based portfolio.
    """

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

    def LocalValueAtRisk(self, confidence_interval: float = 0.99) -> float:
        # Local VAR
        pass

    def ConditionalLocalValueAtRisk(self):
        pass

    def MarginalLocalValueAtRisk(self):
        pass