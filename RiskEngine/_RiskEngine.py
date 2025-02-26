# Global Package Imports
import math

# Local Package Imports
from _utils import (
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
        * portfolio_holdings: historical returns of the asset
        *   NOTE: portfolio_holdings dict must be in the form:
        *         {
        *           asset_1_weight: [asset_1_returns], ...
        *           asset_N_weight: [asset_N_returns]
        *         }
        * market_returns: historical market portfolio returns
        """

        self._portfolio_holdings = portfolio_holdings
        self._market_returns = market_returns

        # Validate the portfolio holdings
        validated = validate_portfolio_holdings(self._portfolio_holdings)

        self._portfolio_returns = portfolio_returns_from_holdings(self._portfolio_holdings)

    @property
    def portfolio_returns(self) -> list:
        """
        * portfolio_returns()
        *
        * Getter function for the portfolio returns
        """

        return self._portfolio_returns

    @property
    def market_returns(self) -> list:
        """
        * market_returns()
        *
        * Getter function for the market returns
        """

        return self._market_returns

    def mean(self, arr: list) -> float:
        """
        * mean()
        *
        * Calculates the mean of a list
        *
        * arr: list of numerical values
        * :returns: mean
        """

        return sum(arr) / len(arr)

    def variance(self, arr: list, sample: bool = True) -> float:
        """
        * variance()
        *
        * Calculates the variance of a list
        *
        * arr: list of numerical values
        * sample: true if sample, false if population
        * :returns: variance
        """

        n = len(arr)

        mu = self.mean(arr)
        return sum((x - mu) ** 2 for x in arr) / (n - 1 if sample else n)

    def standard_deviation(self, arr: list, sample: bool = True) -> float:
        """
        * standard_deviation()
        *
        * Calculates the standard deviation of a list
        *
        * arr: list of numerical values
        * sample: true if sample, false if population
        * :returns: standard deviation
        """

        return math.sqrt(self.variance(arr, sample))

    def skewness(self, arr: list, sample: bool = True) -> float:
        """
        * skewness()
        *
        * Calculates the skew of a list
        *
        * arr: list of numerical values
        * sample: true if sample, false if population
        * :returns: skew
        """

        n = len(arr)

        mu = self.mean(arr)
        sd = self.standard_deviation(arr, sample)

        # Apply Bessel's correction for sample skewness
        bessel = n / ((n - 1)*(n - 2))

        return (bessel if sample else 1) * sum(((x - mu) / sd) ** 3 for x in arr)


    def kurtois(self, arr: list, sample: bool = True) -> float:
        """
        * kurtois()
        *
        * Calculates the kurtois of a list
        *
        * arr: list of numerical values
        * sample: true if sample, false if population
        * :returns: kurtois
        """

        n = len(arr)

        mu = self.mean(arr)
        sd = self.standard_deviation(arr, sample)

        # Always apply the bias factor if a sample
        bias = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))

        # Apply sample bias adjustment
        sample_adj = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        return (bias if sample else 1) * sum(((x - mu) / sd) ** 4 for x in arr) - (sample_adj if sample else 1)