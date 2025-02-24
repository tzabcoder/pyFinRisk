from ._utils import (
    validate_portfolio_holdings,
    portfolio_returns_from_holdings
)

class RiskEngine:
    """
    * RiskEngine
    *
    * The risk engine class sets up the risk analysis framework and directs the
    * computation of any financial risk metric. The risk engine is the base class
    * inherited by the variety of asset-specific risk engines.
    """

    def __init__(self, portfolio_holdings: dict):
        """
        * __init__()
        *
        * Initializes the risk engine class variables.
        *
        * portfolio_holdings: Historical returns of the asset
        *   NOTE: portfolio_holdings dict must be in the form:
        *         {
        *           asset_1_weight: [asset_1_returns], ...
        *           asset_N_weight: [asset_N_returns]
        *         }
        """

        self._portfolio_holdings = portfolio_holdings

        # Validate the portfolio holdings
        validated = utils.validate_portfolio_holdings(self._portfolio_holdings)

        self._portfolio_returns = utils.portfolio_returns_from_holdings(self._portfolio_holdings)

    @property
    def portfolio_returns(self) -> list:
        """
        * portfolio_returns()
        *
        * Getter function for the portfolio returns
        """

        return self._portfolio_returns