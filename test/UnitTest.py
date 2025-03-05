# Global Imports
import yfinance as yf

# Local Imports
from RiskEngine.StockRiskEngine import StockRiskEngine

class UnitTest:
    def __init__(self):
        # Construct the portfolio details
        self.portfolio_details = {
            "Symbols" : ['JPM', 'NVDA', 'LLY'],
            "Shares" : [1000, 2000, 3000],
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

    def BasicPortfolioVAR_NonLog(self, test_counter: int):
        """
        * BasicPortfolioVAR_NonLog()
        *
        * This function tests the basic portfolio VAR calculation.
        * The function uses the test portfolio details and the 99%. 95% and 90% confidence
        * intervals to calculate the VAR. The results are compared to the expected values that
        * were calculated using the 'Portfolio_VAR.xlsx' file.
        """

        test_counter += 1
        print(f"Running Test: BasicPortfolioVAR_NonLog() - Test {test_counter}...")

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
            print(f"90% VAR test failed | Expected: {expected_var_90} | Actual: {_ACTUAL_90_VAR}")
        if not test_95:
            print(f"90% VAR test failed | Expected: {expected_var_95} | Actual: {_ACTUAL_95_VAR}")
        if not test_90:
            print(f"90% VAR test failed | Expected: {expected_var_99} | Actual: {_ACTUAL_99_VAR}")

        if all([test_99, test_95, test_90]):
            # All tests passed
            print("BasicPortfolioVAR_NonLog(): All VAR tests PASSED.")
        else:
            print("BasicPortfolioVAR_NonLog(): FAILED...")

    def BasicPortfolioVAR_Log(self, test_counter: int):
        """
        * BasicPortfolioVAR_Log()
        *
        * This function tests the basic portfolio VAR calculation using log returns.
        * The function uses the test portfolio details and the 99%. 95% and 90% confidence
        * intervals to calculate the VAR. The results are compared to the expected values that
        * were calculated using the 'Portfolio_VAR.xlsx' file.
        """

        test_counter += 1
        print(f"Running Test: BasicPortfolioVAR_Log() - Test {test_counter}...")

        confidence_1 = 0.99
        confidence_2 = 0.95
        confidence_3 = 0.90

        # NOTE: These numbers were caculated manually using the 'Portfolio_VAR.xlsx' file
        _ACTUAL_99_VAR = 0.015644
        _ACTUAL_95_VAR = 0.011078
        _ACTUAL_90_VAR = 0.008594

        # Calculate the VAR at the confidence intervals
        expected_var_99 = self.stock_risk_engine.BasicPortfolioVAR(confidence_interval=confidence_1, log_based=True)
        expected_var_95 = self.stock_risk_engine.BasicPortfolioVAR(confidence_interval=confidence_2, log_based=True)
        expected_var_90 = self.stock_risk_engine.BasicPortfolioVAR(confidence_interval=confidence_3, log_based=True)

        # Check actual vs expected (withing +/- 1% for rounding errors)
        test_99 = abs(expected_var_99 - _ACTUAL_99_VAR) / _ACTUAL_99_VAR < 0.01
        test_95 = abs(expected_var_95 - _ACTUAL_95_VAR) / _ACTUAL_95_VAR < 0.01
        test_90 = abs(expected_var_90 - _ACTUAL_90_VAR) / _ACTUAL_90_VAR < 0.01

        if not test_99:
            print(f"90% VAR test failed | Expected: {expected_var_90} | Actual: {_ACTUAL_90_VAR}")
        if not test_95:
            print(f"90% VAR test failed | Expected: {expected_var_95} | Actual: {_ACTUAL_95_VAR}")
        if not test_90:
            print(f"90% VAR test failed | Expected: {expected_var_99} | Actual: {_ACTUAL_99_VAR}")

        if all([test_99, test_95, test_90]):
            # All tests passed
            print("BasicPortfolioVAR_Log(): All VAR tests PASSED.")
        else:
            print("BasicPortfolioVAR_Log(): FAILED...")

    def BasicPortfolioVAR_Dollar(self, test_counter: int):
        """
        * BasicPortfolioVAR_Dollar()
        *
        * This function tests the basic portfolio VAR calculation using dollar-based VAR.
        * The function uses the test portfolio details and the 99%. 95% and 90% confidence
        * intervals to calculate the VAR. The results are compared to the expected values that
        * were calculated using the 'Portfolio_VAR.xlsx' file.
        """

        test_counter += 1
        print(f"Running Test: BasicPortfolioVAR_Dollar() - Test {test_counter}...")

        confidence_1 = 0.99
        confidence_2 = 0.95
        confidence_3 = 0.90

        # NOTE: These numbers were caculated manually using the 'Portfolio_VAR.xlsx' file
        _ACTUAL_99_VAR = 101303.95
        _ACTUAL_95_VAR = 71738.85
        _ACTUAL_90_VAR = 55651.95

        # Calculate the VAR at the confidence intervals
        expected_var_99 = self.stock_risk_engine.BasicPortfolioVAR(confidence_interval=confidence_1, dollar_based=True)
        expected_var_95 = self.stock_risk_engine.BasicPortfolioVAR(confidence_interval=confidence_2, dollar_based=True)
        expected_var_90 = self.stock_risk_engine.BasicPortfolioVAR(confidence_interval=confidence_3, dollar_based=True)

        # Check actual vs expected (withing +/- 1% for rounding errors)
        test_99 = abs(expected_var_99 - _ACTUAL_99_VAR) / _ACTUAL_99_VAR < 0.01
        test_95 = abs(expected_var_95 - _ACTUAL_95_VAR) / _ACTUAL_95_VAR < 0.01
        test_90 = abs(expected_var_90 - _ACTUAL_90_VAR) / _ACTUAL_90_VAR < 0.01

        if not test_99:
            print(f"90% VAR test failed | Expected: {expected_var_90} | Actual: {_ACTUAL_90_VAR}")
        if not test_95:
            print(f"90% VAR test failed | Expected: {expected_var_95} | Actual: {_ACTUAL_95_VAR}")
        if not test_90:
            print(f"90% VAR test failed | Expected: {expected_var_99} | Actual: {_ACTUAL_99_VAR}")

        if all([test_99, test_95, test_90]):
            # All tests passed
            print("BasicPortfolioVAR_Dollar(): All VAR tests PASSED.")
        else:
            print("BasicPortfolioVAR_Dollar(): FAILED...")

    def run(self):
        """
        * run()
        *
        * Runs the sub-tests for the RiskEngine.
        """

        test_counter = 0

        print('Running Unit Tests...')
        print('------------------------------------------')

        self.BasicPortfolioVAR_NonLog(test_counter)
        self.BasicPortfolioVAR_Log(test_counter)
        self.BasicPortfolioVAR_Dollar(test_counter)