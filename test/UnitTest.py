# Global Imports
import yfinance as yf

# Local Imports
from RiskEngine.StockRiskEngine import StockRiskEngine
from RiskEngine._utils import calculate_returns

class UnitTest:
    def __init__(self):
        # Construct the portfolio details
        self.portfolio_details = {
            "Symbols" : ['JPM', 'NVDA', 'LLY'],
            "Shares" : [1000, 2000, 3000],
        }

        # Download the asset prices
        _prices = []
        self.asset_returns = {}
        for symbol in self.portfolio_details["Symbols"]:
            # Extract the prices
            _temp_prices = yf.download(symbol, start="2023-05-15", end="2024-05-16", auto_adjust=False, progress=False)['Close'][symbol].to_list()
            _prices.append(_temp_prices)

            # Calculate price returns
            self.asset_returns[symbol] = calculate_returns(_temp_prices)
        self.portfolio_details["Prices"] = _prices

        # Download the market prices
        self.market_prices = yf.download('SPY', start="2023-05-15", end="2024-05-16", auto_adjust=False, progress=False)['Close']['SPY'].to_list()
        self.market_returns = calculate_returns(self.market_prices)

        # Initialize the risk engines
        self.stock_risk_engine = StockRiskEngine(self.portfolio_details, self.market_prices)

    def Beta(self, test_counter: int) -> int:
        """
        * Beta()
        *
        * This function tests the beta calculation.
        * The results are compared to the expected values that were caculated using
        * the 'Portfolio_VAR.xlsx' file.
        """

        test_counter += 1
        print(f"Running Test: Beta() - Test {test_counter}...")

        # NOTE: These numbers were caculated manually using the 'Portfolio_VAR.xlsx' file
        expected_betas = {
            "JPM" : 0.717344664,
            "NVDA" : 2.384378483,
            "LLY" : 0.599788671
        }

        tests = []

        # Calculate the beta for each stock
        for s, r in self.asset_returns.items():
            actual_beta = self.stock_risk_engine.Beta(r, self.market_returns)

            # Compare the actual beta vs the expected beta (within +/- 1% for rounding errors)
            test_beta = abs(actual_beta - expected_betas[s]) / expected_betas[s] < 0.01
            tests.append(test_beta)

            if not test_beta:
                print(f"{s} Beta test failed | Expected: {expected_betas[s]} | Actual: {actual_beta}")

        if all(tests):
            print("Beta(): All beta tests PASSED.")
        else:
            print("Beta(): FAILED...")

        return test_counter

    def BasicPortfolioVAR_NonLog(self, test_counter: int) -> int:
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

        # Check actual vs expected (within +/- 1% for rounding errors)
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

        return test_counter

    def BasicPortfolioVAR_Log(self, test_counter: int) -> int:
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

        # Check actual vs expected (within +/- 1% for rounding errors)
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

        return test_counter

    def BasicPortfolioVAR_Dollar(self, test_counter: int) -> int:
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

        # Check actual vs expected (within +/- 1% for rounding errors)
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

        return test_counter

    def DisplayPortfolioStatistics(self, test_counter: int) -> int:
        """
        * DisplayPortfolioStatistics()
        *
        * This function tests the display of the portfolio statistics. This test
        * always passes and is used to visually inspect the portfolio statistics.
        """

        test_counter += 1
        print(f"Running Test: DisplayPortfolioStatistics() - Test {test_counter}...")

        # Display the portfolio statistics
        self.stock_risk_engine.DisplayPortfolioStatistics(plot=True)

        return test_counter

    def run(self):
        """
        * run()
        *
        * Runs the sub-tests for the RiskEngine.
        """

        test_counter = 0

        print('Running Unit Tests...')
        print('------------------------------------------')

        test_counter = self.Beta(test_counter)
        test_counter = self.BasicPortfolioVAR_NonLog(test_counter)
        test_counter = self.BasicPortfolioVAR_Log(test_counter)
        test_counter = self.BasicPortfolioVAR_Dollar(test_counter)
        test_counter = self.DisplayPortfolioStatistics(test_counter)

        print('------------------------------------------')
        print('Unit Tests Complete.')

        # TODO: Create a result summary