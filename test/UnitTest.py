# Global Imports
import yfinance as yf

# Local Imports
from RiskEngine.StockRiskEngine import StockRiskEngine

class UnitTest:
    def __init__(self):
        # Construct the portfolio details
        self.portfolio_details = {
            "Symbols" : ['JPM', 'NVDA', 'LLY'],
            "Weights" : [0.167, 0.333, 0.500],
        }

        # Download the asset prices
        _prices = []
        for symbol in self.portfolio_details["Symbols"]:
            _prices.append(yf.download(symbol, start="2023-05-15", end="2024-05-16", auto_adjust=False, progress=False)['Close'][symbol].to_list())
        self.portfolio_details["Prices"] = _prices

        # Download the market prices
        self.market_prices = yf.download('SPY', start="2023-05-15", end="2024-05-16", auto_adjust=False, progress=False)['Close']['SPY'].to_list()

        # Initialize the risk engines
        self.stock_risk_engine = StockRiskEngine(self.portfolio_details, self.market_prices)

    def _basic_portfolio_var(self):
        """
        * _basic_portfolio_var()
        *
        * This function tests the basic portfolio VAR calculation.
        * The function uses the test portfolio details and the 99%. 95% and 90% confidence
        * intervals to calculate the VAR. The results are compared to the expected values that
        * were calculated using the 'Portfolio_VAR.xlsx' file.
        """

        confidence_1 = 0.99
        confidence_2 = 0.95
        confidence_3 = 0.90

        # NOTE: These numbers were caculated manually using the 'Portfolio_VAR.xlsx' file
        _ACTUAL_99_VAR = 0.036805
        _ACTUAL_95_VAR = 0.026064
        _ACTUAL_90_VAR = 0.020219

        # Calculate the VAR at the confidence intervals
        expected_var_99 = self.stock_risk_engine.BasicPortfolioVAR(confidence_interval=confidence_1)
        expected_var_95 = self.stock_risk_engine.BasicPortfolioVAR(confidence_interval=confidence_2)
        expected_var_90 = self.stock_risk_engine.BasicPortfolioVAR(confidence_interval=confidence_3)

        # Check actual vs expected (withing +/- 1% for rounding errors)
        test_99 = abs(expected_var_99 - _ACTUAL_99_VAR) / _ACTUAL_99_VAR < 0.01
        test_95 = abs(expected_var_95 - _ACTUAL_95_VAR) / _ACTUAL_95_VAR < 0.01
        test_90 = abs(expected_var_90 - _ACTUAL_90_VAR) / _ACTUAL_90_VAR < 0.01

        if not test_99:
            print("99% VAR test failed...")
        if not test_95:
            print("95% VAR test failed...")
        if not test_90:
            print("90% VAR test failed...")

        if all([test_99, test_95, test_90]):
            # All tests passed
            print("BasicPortfolioVAR(): All VAR tests PASSED.")
        else:
            print("BasicPortfolioVAR(): FAILED...")

    def run(self):
        print('Running Unit Tests...')
        print('------------------------------------------')

        self._basic_portfolio_var()