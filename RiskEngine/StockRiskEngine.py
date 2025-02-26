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

    def __init__(self, portfolio_holdings: dict, market_returns: list):
        """
        * __init__()
        *
        * Initializes the stock risk engine and the base class.
        *
        * portfolio_holdings: historical returns of the asset
        *   NOTE: portfolio_holdings dict must be in the form:
        *         {
        *           (asset_1_symbol, asset_1_weight): [asset_1_returns], ...
        *           (asset_N_symbol, asset_N_weight): [asset_N_returns]
        *         }
        * market_returns: historical returns of the market portfolio (benchmark)
        *   NOTE: the market returns must be at least as long as the portfolio returns
        """

        RiskEngine.__init__(self, portfolio_holdings, market_returns)

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

    def LocalValueAtRisk():
        pass





# TEST code. Formal UTs will be created
holdings = {
    0.5: [0.05, 0.05, 0.05],
    0.25: [0.025, 0.025, 0.025],
    0.25: [0.01, 0.01, 0.01]
}

market_returns = [0.12, 0.12, 0.12]

risk_engine = StockRiskEngine(holdings)

beta = risk_engine.beta(market_returns)
print(beta)