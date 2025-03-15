# Global Package Imports
import math
import pandas as pd
import scipy.stats as stats

# Local Package Imports
from RiskEngine._utils import (
    calculate_returns,
    calculate_asset_returns,
    calculate_asset_weights,
    calculate_returns_from_holdings
)

class RiskEngine:
    """
    * RiskEngine
    *
    * The risk engine class sets up the risk analysis framework and directs the
    * computation of financial risk metrics. The risk engine is the base class
    * inherited by the variety of asset-specific risk engines. It primarily is an
    * accessor for portfolio components and calculates statistical properties.
    """

    def __init__(self, portfolio_details: dict, market_prices: list):
        """
        * __init__()
        *
        * Initializes the stock risk engine and the base class.
        *
        * portfolio_details: details of the financial portfolio (symbols, shares, prices)
        *                    each dict value index is matched accross all keys, it is assumed that
        *                    prices are DAILY prices.
        *   NOTE: portfolio_details dict must be in the form:
        *         {
        *           "Symbols" : [Symbol_1, Symbol_2, ..., Symbol_N],
        *           "Shares" : [Shares_1, Shares_2, ..., Shares_N],
        *           "Prices" : [[Prices_1], [Prices_2], ..., [Prices_N]]
        *         }
        * market_prices: historical DAILY prices of the market portfolio (benchmark)
        * NOTE: the market returns must be at least as long as the portfolio returns
        """

        self._portfolio_details = portfolio_details
        self._market_prices  = market_prices

        # Adds _portfolio_details['Weights']
        calculate_asset_weights(self._portfolio_details)

        # Adds _portfolio_details['Returns']
        calculate_asset_returns(self._portfolio_details)

        # Calculate the portfolio and market returns
        self._portfolio_returns = calculate_returns_from_holdings(self._portfolio_details)
        self._market_returns = calculate_returns(self._market_prices)

    ####################################################################
    # Getter Properties
    ####################################################################
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
    def portfolio_shares(self) -> list:
        return self._portfolio_details['Shares']

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
    # Portfolio Accessor Functions
    ####################################################################
    def get_asset_returns(self, symbol: str) -> list:
        """
        * get_asset_returns()
        *
        * Returns the asset returns for a given symbol.
        *
        * symbol: the symbol of the asset
        * :returns: the asset returns for the given symbol
        """

        if symbol in self.portfolio_symbols:
            index = self.portfolio_symbols.index(symbol)
            return self.portfolio_asset_returns[index]
        else:
            raise ValueError(f"Symbol {symbol} not found in portfolio...")

    def get_asset_prices(self, symbol: str) -> list:
        """
        * get_asset_price()
        *
        * Returns the asset prices for a given symbol.
        *
        * symbol: the symbol of the asset
        * :returns: the asset prices for the given symbol
        """

        if symbol in self.portfolio_symbols:
            index = self.portfolio_symbols.index(symbol)
            return self.portfolio_prices[index]
        else:
            raise ValueError(f"Symbol {symbol} not found in portfolio...")

    def get_asset_weight(self, symbol: str) -> float:
        """
        * get_asset_weight()
        *
        * Returns the asset weight for a given symbol.
        *
        * symbol: the symbol of the asset
        * :returns: the asset weight for the given symbol
        """

        if symbol in self.portfolio_symbols:
            index = self.portfolio_symbols.index(symbol)
            return self.portfolio_weights[index]
        else:
            raise ValueError(f"Symbol {symbol} not found in portfolio...")

    def get_asset_shares(self, symbol: str) -> float:
        """
        * get_asset_shares()
        *
        * Returns the asset shares for a given symbol.
        *
        * symbol: the symbol of the asset
        * :returns: the asset shares for the given symbol
        """

        if symbol in self.portfolio_symbols:
            index = self.portfolio_symbols.index(symbol)
            return self._portfolio_details['Shares'][index]
        else:
            raise ValueError(f"Symbol {symbol} not found in portfolio...")

    ####################################################################
    # Statistical Functions
    ####################################################################
    def covariance(self, arr1: list, arr2: list, sample: bool = True) -> float:
        """
        * covariance()
        *
        * Calculates the covariance between two lists: cov(arr1, arr2).
        *
        * arr1: list of numerical values
        * arr2: list of numerical values
        * sample: true if sample, false if population
        * :returns: covariance coefficient
        """

        # Return the covariance coefficient
        return np.cov(arr1, arr2, ddof=1 if sample else 0)[0][1]

    def critical_z_score(self, confidence_interval: float) -> float:
        """
        * critical_z_score()
        *
        * Calculates the critical z-score for a given confidence interval
        * NOTE: The confidence interval is already passed as 1 - alpha
        *
        * confidence_interval: confidence interval for the z-score calculation
        * :returns: the critical z-score
        """

        # Get the CRITICAL z-score from the confidence interval (right-tailed test)
        # The confidence interval is already passed as 1 - alpha, where alpha is the significance level
        z_score = stats.norm.ppf(confidence_interval)

        return z_score

    def mean(self, arr: list) -> float:
        """
        * mean()
        *
        * Calculates the mean of a list.
        *
        * arr: list of numerical values
        * :returns: mean
        """

        return sum(arr) / len(arr)

    def variance(self, arr: list, sample: bool = True) -> float:
        """
        * variance()
        *
        * Calculates the variance of a list.
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
        * Calculates the standard deviation of a list.
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
        * Calculates the skew of a list.
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


    def kurtosis(self, arr: list, sample: bool = True) -> float:
        """
        * kurtosis()
        *
        * Calculates the kurtosis of a list.
        *
        * arr: list of numerical values
        * sample: true if sample, false if population
        * :returns: kurtosis
        """

        n = len(arr)

        mu = self.mean(arr)
        sd = self.standard_deviation(arr, sample)

        # Always apply the bias factor if a sample
        bias = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))

        # Apply sample bias adjustment
        sample_adj = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        return (bias if sample else 1) * sum(((x - mu) / sd) ** 4 for x in arr) - (sample_adj if sample else 1)