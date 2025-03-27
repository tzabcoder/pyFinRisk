# Global Package Imports
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels import regression

# Local Package Imports
from RiskEngine._RiskEngine import RiskEngine
from RiskEngine._utils import (
    calculate_portfolio_value
)

# Settings
plt.style.use('ggplot')

class StockRiskEngine(RiskEngine):
    """
    * StockRiskEngine(RiskEngine)
    *
    * The stock risk engine is the framework for calculating all financial risk
    * metrics associated with an equity-based portfolio. Each component of the financial
    * portfolio is considered a risk factor.
    """

    def __init__(self, portfolio_details: dict, market_prices: list) -> None:
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

    def PortfolioStatistics(self, display: bool = False, plot: bool = False) -> dict:
        """
        * PortfolioStatistics()
        *
        * Get the first-order portfolio statistics relating to the return distribution.
        * The first-order statistics describe the historical return distribution of the
        * portfolio.
        * If the display flag is true, the function will display the portfolio statistics.
        * If the plot flag is true, the function will plot the historical portfolio
        * returns vs. the market and the distribution of historical returns.
        *
        * plot: flag to plot the portfolio statistics
        * :return: None
        """

        # Calculate the first-order portfolio statistics
        statistics = {
            'mean' :               self.mean(self.portfolio_returns),
            'variance' :           self.variance(self.portfolio_returns),
            'standard_deviation' : self.standard_deviation(self.portfolio_returns),
            'skew' :               self.skewness(self.portfolio_returns),
            'kurtosis' :           self.kurtosis(self.portfolio_returns),
            'beta' :               self.Beta(self.portfolio_returns, self.market_returns)
        }

        # Display the portfolio statistics
        if display:
            print("Portfolio Statistics")
            print('==================================================')
            print(f"Beta of Portfolio:  -------------------  {statistics['beta']:.4f}")
            print(f"Mean of Portfolio Returns:  -----------  {(statistics['mean'] * 100):.4f}%")
            print(f"Variance of Portfolio Returns:  -------  {(statistics['variance'] * 100):.4f}%")
            print(f"Standard Deviation of Portfolio Returns: {(statistics['standard_deviation'] * 100):.4f}%")
            print(f"Skewness of Portfolio Returns:  -------  {statistics['skew']:.4f}")
            print(f"Kurtosis of Portfolio Returns:  -------  {statistics['kurtosis']:.4f}")
            print('==================================================')

        # Plot the portfolio returns and distributios
        if plot:
            fig, axes = plt.subplots(2, 2, figsize = (18, 12))

            # Plot the historical returns of the portfolio ====================
            cummulative_portfolio_returns = np.cumsum(self.portfolio_returns)
            cummulative_market_returns = np.cumsum(self.market_returns)

            axes[0][0].plot(cummulative_portfolio_returns, label="Portfolio Returns")
            axes[0][0].plot(cummulative_market_returns, label="Market Returns")
            axes[0][0].legend()
            axes[0][0].set_title("Portfolio Returns")
            axes[0][0].set_xlabel("Days")
            axes[0][0].set_ylabel("Returns")

            axes[0][1].plot(self.portfolio_returns, label="Daily Portfolio Returns")
            axes[0][1].legend()
            axes[0][1].set_title("Daily Portfolio Returns")
            axes[0][1].set_xlabel("Days")
            axes[0][1].set_ylabel("Returns")

            # Plot the historical return distribution ====================
            var = self.PortfolioVAR(confidence_interval=0.01)
            axes[1][0].hist(self.portfolio_returns, bins = 50, alpha = 0.7, density = True, label="Portfolio Returns")
            axes[1][0].axvline(x=var, color='cyan', label='99% VaR')
            axes[1][0].axvline(x=np.mean(self.portfolio_returns), color="black", alpha=1, label='Mean')
            axes[1][0].legend()
            axes[1][0].set_title('Portfolio Returns Distribution')
            axes[1][0].set_xlabel('Daily Returns')
            axes[1][0].set_ylabel('Frequency')

            mu, std = norm.fit(self.portfolio_returns)
            xmin, xmax = axes[1][0].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            axes[1][0].plot(x, p, 'k', linewidth=0.5, label='Daily Return Distribution')

            # Plot the cummulative distribution function (CDF) ====================
            var_values = []
            ci_values = []

            start = 0
            stop = 1
            increment = 0.05
            i = start
            while i <= stop:
                var_values.append(self.PortfolioVAR(confidence_interval=i))
                ci_values.append(i)

                i += increment

            axes[1][1].plot(var_values, ci_values)
            axes[1][1].set_title("VaR Cummulative Distribution")
            axes[1][1].set_xlabel("VaR (%)")
            axes[1][1].set_ylabel("Confidence Interval")

            # Show the plot
            plt.show()

        return statistics

    ####################################################################
    # Historical Value at Risk Functions
    ####################################################################
    def IndividualVAR(self, symbol: str, confidence_interval: float = 0.01) -> float:
        """
        * IndividualVAR()
        *
        * Calculates the individual value at risk (VAR) for an asset using the historical
        * asset returns.
        * The individual VAR is calculated as:
        *  VAR = sd * z * w
        *
        * confidence_interval: confidence interval for the VAR calculation
        * :returns: the individual VAR, None if failure
        """

        # Validate confidence interval
        if confidence_interval < 0 or confidence_interval > 1:
            raise ValueError('Confidence interval must be greater than or equal to 0 and less than or equal to 1...')

        # Get the asset returns
        asset_return = self.get_asset_returns(symbol)

        # Get the asset weight
        asset_weight = self.get_asset_weight(symbol)

        # Calculate the individual portfolio risk
        stddev = self.standard_deviation(asset_return)

        var = stddev * self.critical_z_score(confidence_interval) * asset_weight

        return var

    def PortfolioVAR(self, confidence_interval: float = 0.01, dollar_based: bool = False) -> float:
        """
        * PortfolioVAR()
        *
        * Calculates the basic historical portfolio value at risk (VAR) using the covariance
        * matrix of the portfolio's assets. This method calculates VAR based on its
        * historical asset returns.
        * The portfolio variance is calculated as:
        *  V = w * COV * w'
        * The portfolio risk is then calculated as:
        *  R = sqrt(V)
        * The VAR is then calculated as:
        *  VAR = R * z
        * where z is the critical z-score of the confidence interval.
        *
        * confidence_interval: confidence interval for the VAR calculation
        * dollar_based: if True, use dollar based VAR, else use percentage based VAR
        * :returns: the basic portfolio VAR, None if failure
        """

        # Validate confidence interval
        if confidence_interval < 0 or confidence_interval > 1:
            raise ValueError('Confidence interval must be greater than or equal to 0 and less than or equal to 1...')

        # Calculate the covariance matrix
        cov_matrix = np.cov(self.portfolio_asset_returns, rowvar = True)

        # Variance of portfolio rate of return
        portfolio_variance = self.portfolio_weights @ cov_matrix @ np.transpose(self.portfolio_weights)

        # Calculate the portfolio risk
        portfolio_risk = math.sqrt(portfolio_variance)

        # Calculate and return the basic VAR
        var = portfolio_risk * self.critical_z_score(confidence_interval)

        if dollar_based:
            return var * calculate_portfolio_value(self.portfolio_shares, self.portfolio_prices)
        else:
            return var

    def SimulatedPortfolioVAR(self, sims: int, n: int, confidence_interval: float = 0.01) -> float:
        """
        * SimulatedPortfolioVAR()
        *
        * Simulates the portfolio VAR over the next n days using Monte Carlo sims and Geometric
        * Brownian Motion (GBM). The portfolio VAR is calculated a number of times and averaged
        * to obtain the forecasted VAR over the next n days.
        *
        * sims: number of times to simulate the portfolio VAR
        * n: number of days to estimate one occurence of portfolio VAR
        * confidence_interval: confidence interval for the VAR calculation
        * dollar_based: if True, return the portfolio VAR change in dollars, else return in percentage
        * :returns: the simulated dollar-based portfolio VAR over the next n days
        """

        simulated_vars = []

        # Run all simulations
        for i in range(sims):
            p_var = self.simulate_returns(n)

            # Calculate the VAR threshold
            observations = len(p_var)
            threshold = int(observations * (1 - confidence_interval))

            simulated_vars.append(sorted(p_var, reverse=True)[threshold])

        return sum(simulated_vars) / len(simulated_vars)

    def ConditionalVAR(self, confidence_interval: float = 0.01, dollar_based: bool = False) -> float:
        """
        * ConditionalVAR()
        *
        * Calculates the conditional value at risk (CVAR) for the portfolio.
        * CVAR measures the expected loss that exceeds the VAR's threshold, and provides
        * an estimate of the average loss in extreme scenarios beyond the VAR limit.
        *
        * confidence_interval: confidence interval for the VAR calculation
        * dollar_based: if True, return the portfolio VAR change in dollars, else return in percentage
        * :returns: the conditional VAR (in absolute return terms), None if failure
        """

        # Sort the portfolio returns from smallest to largest
        portfolio_returns = sorted(self.portfolio_returns)

        # Calculate the VAR threshold
        observations = len(portfolio_returns)
        threshold = int(observations * (1 - confidence_interval))

        # Calculate conditional VaR (CVaR)
        c_var = portfolio_returns[threshold]

        if dollar_based:
            return c_var * calculate_portfolio_value(self.portfolio_shares, self.portfolio_prices)
        else:
            return c_var

    def MarginalLocalVAR(self, symbol: str, confidence_interval: float = 0.01, dollar_based: bool = False) -> float:
        """
        * MarginalLocalVAR()
        *
        * Calculates the marginal local value at risk (MVAR) for an asset using the
        * portfolio's return standard deviations, beta of the asset's returns, and the
        * risk of the portfolio.
        * The marginal VAR calculates the change in portfolio VAR when an asset's weight changes marginally.
        *
        * Ex: If the MVaR for the asset is 2%, then the portfolio VaR will increase by 2% if the
        *     asset's weight increases marginally.
        *
        * symbol: symbol of the asset to calculate the marginal VAR
        * confidence_interval: confidence interval for the VAR calculation
        * dollar_based: if True, return the portfolio VAR change in dollars, else return in percentage
        * :returns: the marginal local VAR (in absolute return terms), None if failure
        """

        # Validate the confidence interval
        if confidence_interval < 0 or confidence_interval > 1:
            raise ValueError('Confidence interval must be greater than or equal to 0 and less than or equal to 1...')

        # Calculate the CRITICAL z-score
        z_score = self.critical_z_score(confidence_interval)

        # Calculate the portfolio VAR
        portfolio_var = self.PortfolioVAR(confidence_interval, dollar_based = False)

        beta = self.Beta(self.get_asset_returns(symbol), self.market_returns)

        # Calculate the marginal VAR
        m_var = beta * portfolio_var

        if dollar_based:
            return m_var * calculate_portfolio_value(self.portfolio_shares, self.portfolio_prices)
        else:
            return m_var

    def IncrementalLocalVAR(self, symbol: str, weight_change: float, confidence_interval: float = 0.01, dollar_based: bool = False) -> float:
        """
        * IncrementalLocalVAR()
        *
        * Calculates the incremental local value at risk (IVAR) for an asset using the new
        * weight of the asset and the marginal value at risk (MVAR).
        * The incremental VAR approximates the change in portfolio VAR when an asset's weight changes
        * significantly.
        *
        * Ex: If the IVaR for the asset is 1%, then the approximate impact on the portfolio VaR by
        *     changing the risk factor's weight by N% is N% * 1% assuming a linear approximation for
        *     small changes.
        *
        * symbol: symbol of the asset to calculate the incremental VAR
        * weight_change: change in weight of the position (+/-)
        * confidence_interval: confidence interval for the VAR calculation
        * dollar_based: if True, return the portfolio VAR change in dollars, else return in percentage
        * :returns: the incremental local VAR (in absolute return terms), None if failure
        """

        # Validate the confidence interval
        if confidence_interval < 0 or confidence_interval > 1:
            raise ValueError('Confidence interval must be greater than or equal to 0 and less than or equal to 1...')

        # Calculate the marginal VaR for the position
        m_var = self.MarginalLocalVAR(symbol, confidence_interval)

        # Calculate the incremental VAR
        i_var = m_var * weight_change

        if dollar_based:
            return i_var * calculate_portfolio_value(self.portfolio_shares, self.portfolio_prices)
        else:
            return i_var

    def ComponentLocalVAR(self, symbol: str, confidence_interval: float = 0.01, dollar_based: bool = False) -> float:
        """
        * ComponentLocalVAR()
        *
        * Calculates the components local value at risk (VAR) for an asset using the VAR,
        * the beta of the risk factor, and the weight of the risk factor.
        *
        * The component VaR approximates how much the portfolio VaR would change if the component
        * was removed from the portfolio.
        *
        * symbol: symbol of the asset to calculate the component VAR
        * confidence_interval: confidence interval for the VAR calculation
        * dollar_based: if True, return the portfolio VAR change in dollars, else return in percentage
        * :returns: the component local VAR (in absolute return terms), None if failure
        """

        # Validate the confidence interval
        if confidence_interval < 0 or confidence_interval > 1:
            raise ValueError('Confidence interval must be greater than or equal to 0 and less than or equal to 1...')

        # Calculate the component's beta
        beta = self.Beta(self.get_asset_returns(symbol), self.market_returns)
        weight = self.get_asset_weight(symbol)

        # Calculate the component VaR
        c_var = self.PortfolioVAR(confidence_interval) * beta * weight

        self.simulate_returns(1000)

        if dollar_based:
            return c_var * calculate_portfolio_value(self.portfolio_shares, self.portfolio_prices)
        else:
            return c_var