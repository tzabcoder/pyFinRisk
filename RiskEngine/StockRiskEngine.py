# Global Package Imports
import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels import regression

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

    ####################################################################
    # Support Functions
    ####################################################################
    def Beta(self, returns: list, market_returns: list) -> float:
        """
        * Beta()
        *
        * Linear, first-order risk metric for stocks.
        * Beta measures the assets's return sensitivity to the benchmark.
        * Calculated by finding the slope of the regression between the asset
        * and market returns.
        * NOTE: The size of the market returns MUST be the same periocidy of the
        *       asset and have at least as many occurances as the asset.
        *
        * returns: list of returns to calculate the beta
        * market_returns: list of market returns
        * :returns: beta of the asset, None if failure
        """

        beta = None
        N = len(returns)

        if len(market_returns) < N:
            raise ValueError("Insufficient market returns...")

        else:
            # Format the returns
            market_returns = market_returns[:N]
            X = market_returns
            y = returns

            x = sm.add_constant(X)
            model = regression.linear_model.OLS(y, x).fit()

            # Remove the constant term
            x = x[:, 1]

            # Calculate the beta coefficient
            beta = model.params[1]

        return beta

    def DisplayPortfolioStatistics(self, plot: bool = False) -> None:
        """
        * DisplayPortfolioStatistics()
        *
        * Displays the first-order portfolio statistics relating to the return distribution.
        * The first-order statistics describe the historical return distribution of the
        * portfolio. If the plot flag is true, the function will plot the historical portfolio
        * returns vs. the market and the distribution of historical returns.
        *
        * plot: flag to plot the portfolio statistics
        """

        # Calculate the portfolio statistics
        mean = self.mean(self.portfolio_returns)
        variance = self.variance(self.portfolio_returns)
        standard_deviation = self.standard_deviation(self.portfolio_returns)
        skewness = self.skewness(self.portfolio_returns)
        kurtosis = self.kurtosis(self.portfolio_returns)
        beta = self.Beta(self.portfolio_returns, self.market_returns)

        # Display the portfolio statistics
        print("Portfolio Statistics")
        print('==================================================')
        print(f"Beta of Portfolio:  -------------------  {beta:.4f}")
        print(f"Mean of Portfolio Returns:  -----------  {(mean * 100):.4f}%")
        print(f"Variance of Portfolio Returns:  -------  {(variance * 100):.4f}%")
        print(f"Standard Deviation of Portfolio Returns: {(standard_deviation * 100):.4f}%")
        print(f"Skewness of Portfolio Returns:  -------  {skewness:.4f}")
        print(f"Kurtosis of Portfolio Returns:  -------  {kurtosis:.4f}")
        print('==================================================')

        # Plot the portfolio returns and distribution
        if plot:
            fig, axes = plt.subplots(1, 2, figsize = (15, 5))

            # Plot the historical returns of the portfolio
            cummulative_portfolio_returns = np.cumsum(self.portfolio_returns)
            cummulative_market_returns = np.cumsum(self.market_returns)

            axes[0].plot(cummulative_portfolio_returns, label="Portfolio Returns")
            axes[0].plot(cummulative_market_returns, label="Market Returns")
            axes[0].legend()
            axes[0].set_title("Portfolio Returns")
            axes[0].set_xlabel("Days")
            axes[0].set_ylabel("Returns")

            # Plot the historical return distribution
            axes[1].hist(self.portfolio_returns, bins = 50, edgecolor = 'black', alpha = 0.7)
            axes[1].set_title('Portfolio Returns Distribution')
            axes[1].set_xlabel('Returns')
            axes[1].set_ylabel('Frequency')

            # Show the plot
            plt.show()

    ####################################################################
    # Value at Risk Functions
    ####################################################################
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

    def MarginalLocalVAR(self, symbol: str) -> float:
        """
        * MarginalLocalVAR()
        *
        * Calculates the marginal local value at risk (VAR) for an asset using the
        * basic VAR of the portfolio and the beta of the asset. The Marginal VAR measures
        * the change in portfolio VAR as a result from taking additional dollar exposure
        * to a given component.
        *
        * symbol: symbol of the asset to calculate the marginal VAR
        * :returns: the marginal local VAR (in dollar terms), None if failure
        """

        asset_return = self.get_asset_returns(symbol)

        # Calculate the beta of the individual position
        beta_i = self.Beta(asset_return, self.market_returns)

        # Calculate the dollar basic VAR of the portfolio
        basic_var = self.BasicPortfolioVAR(dollar_based=True)

        if beta_i is None or basic_var is None:
            marginal_var = None

        else:
            marginal_var = (basic_var / self.calculate_portfolio_value(self.portfolio_shares, self.portfolio_prices)) * beta_i

        # Returns the marginal VAR for the asset
        return marginal_var

    def ConditionalLocalVAR(self):
        pass