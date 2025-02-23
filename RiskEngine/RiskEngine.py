
class RiskEngine:
    """
    * RiskEngine
    *
    * The risk engine class sets up the risk analysis framework and directs the
    * computation of any financial risk metric. The engine accepts a variety of portfolio
    * types.
    """

    VALID_PORTFOLIO_TYPES = ['Stock', 'Bond', 'Derivatives']

    def __init__(self, portfolio_holdings: dict, portfolio_type: str):
        """
        * __init__()
        *
        * Initializes the risk engine class variables.
        *
        * portfolio_holdings: Historical returns of the asset
        *   NOTE: portfolio_holdings dict must be in the form:
        *         {
        *           'asset_1': [asset_1_returns], ...
        *           'asset_N': [asset_N_returns]
        *         }
        * portfolio_type: Type of the portfolio to calculate risk metrics (Stock, Bond, Derivative)
        """

        self._portfolio_holdings = portfolio_holdings

        if portfolio_type in self.VALID_PORTFOLIO_TYPES:
            self._portfolio_type = portfolio_type
        else:
            raise ValueError(f'portfolio_type must be one of: {self.VALID_PORTFOLIO_TYPES}')