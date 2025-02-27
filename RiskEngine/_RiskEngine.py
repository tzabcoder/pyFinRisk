# Global Package Imports
import math

# Local Package Imports
from _utils import (
    calculate_returns,
    calculate_asset_returns,
    calculate_returns_from_holdings
)

class RiskEngine:
    """
    * RiskEngine
    *
    * The risk engine class sets up the risk analysis framework and directs the
    * computation of any financial risk metric. The risk engine is the base class
    * inherited by the variety of asset-specific risk engines.
    """

    def __init__(self, portfolio_details: dict, market_prices: list):
        """
        * __init__()
        *
        * Initializes the risk engine class variables.
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
        """

        self._portfolio_details = portfolio_details
        self._market_prices  = market_prices

        # Adds _portfolio_details['Returns']
        calculate_asset_returns(self._portfolio_details)

        # Calculate the log-based daily returns
        self._portfolio_returns = calculate_returns_from_holdings(self._portfolio_returns)
        self._market_returns = calculate_returns(self._market_prices)

    ####################################################################
    # Getter properties
    @property
    def portfolio_returns(self) -> list:
        return self._portfolio_returns

    @property
    def market_returns(self) -> list:
        return self._market_returns

    # Portfolio details getter properties
    @property
    def portfolio_symbols(self) -> list:
        return self._portfolio_details['Symbols']

    @property
    def portfolio_weights(self) -> list:
        return self._portfolio_details['Weights']

    @property
    def portfolio_prices(self) -> list:
        return self._portfolio_details['Prices']

    @property
    def portfolio_asset_returns(self) -> list:
        if 'Returns' in self._portfolio_details:
            return self._portfolio_details['Returns']
        else:
            return None
    ####################################################################

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