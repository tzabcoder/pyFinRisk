from RiskEngine import RiskEngine

class StockRiskEngine(RiskEngine):
    """
    * StockRiskEngine(RiskEngine)
    *
    * The stock risk engine is the framework for calculating all financial risk
    * metrics associated with an equity-based portfolio.
    """

    def __init__(self, portfolio_holdings: dict):
        print('StockRiskEngine')
        RiskEngine.__init__(self, portfolio_holdings)

    def beta(self, market_returns: list) -> int:
        N = len(self.portfolio_returns())

        if len(market_returns) > N:
            raise ValueError("")

        return 0