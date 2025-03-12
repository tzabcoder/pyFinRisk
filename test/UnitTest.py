# Global Imports
import math
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
            print("Beta(): All beta tests PASSED.\n")
            return True
        else:
            print("Beta(): FAILED...\n")
            return False

    def IndividualVAR(self, test_counter: int) -> int:
        """
        * IndividualVAR()
        *
        * This function tests the basic individual VAR calculation.
        * The function uses the test portfolio details and the 99%. 95% and 90% confidence
        * intervals to calculate the each asset's VAR. The results are compared to the expected values that
        * were calculated using the 'Portfolio_VAR.xlsx' file.
        """

        print(f"Running Test: IndividualVAR() - Test {test_counter}...")

        confidence_intervals = [0.99, 0.95, 0.90]

        # NOTE: These numbers were caculated manually using the 'Portfolio_VAR.xlsx' file
        # _###_VAR[0] is at the 99% confidence interval
        # _###_VAR[1] is at the 95% confidence interval
        # _###_VAR[2] is at the 90% confidence interval
        _EXPECTED_VALUES = {
            "JPM" : [0.004048, 0.002867, 0.002224],
            "NVDA" : [0.024213, 0.017147, 0.013302],
            "LLY" : [0.021952, 0.015546, 0.012060]
        }

        all_tests = True

        for i in range(len(self.portfolio_details['Symbols'])):
            symbol = self.portfolio_details['Symbols'][i]
            ci = confidence_intervals[i]

            # Calculate the marginal VAR
            actual_individual_var = self.stock_risk_engine.IndividualVAR(symbol, ci)

            # Check actual vs expected (within +/- 1% for rounding errors)
            test = math.isclose(actual_individual_var, _EXPECTED_VALUES[symbol][i], rel_tol=0.01)

            if not test:
                print(f"{ci*100}% Individual VAR test failed | Actual: {actual_individual_var} | Expected: {_EXPECTED_VALUES[symbol][i]}")
                all_tests = False

        if all_tests:
            print("IndividualVAR(): All VAR tests PASSED.\n")
            return True
        else:
            print("IndividualVAR(): FAILED...\n")
            return False

    def PortfolioVAR(self, test_counter: int) -> int:
        """
        * PortfolioVAR()
        *
        * This function tests the basic portfolio VAR calculation.
        * The function uses the test portfolio details and the 99%. 95% and 90% confidence
        * intervals to calculate the VAR. The results are compared to the expected values that
        * were calculated using the 'Portfolio_VAR.xlsx' file.
        """

        print(f"Running Test: PortfolioVAR() - Test {test_counter}...")

        confidence_1 = 0.99
        confidence_2 = 0.95
        confidence_3 = 0.90

        # NOTE: These numbers were caculated manually using the 'Portfolio_VAR.xlsx' file
        _EXPECTED_99_VAR = 0.036805
        _EXPECTED_95_VAR = 0.026064
        _EXPECTED_90_VAR = 0.020219

        # Calculate the VAR at the confidence intervals
        actual_var_99 = self.stock_risk_engine.PortfolioVAR(confidence_interval=confidence_1)
        actual_var_95 = self.stock_risk_engine.PortfolioVAR(confidence_interval=confidence_2)
        actual_var_90 = self.stock_risk_engine.PortfolioVAR(confidence_interval=confidence_3)

        # Check actual vs expected (within +/- 1% for rounding errors)
        test_99 = math.isclose(actual_var_99, _EXPECTED_99_VAR, rel_tol=0.01)
        test_95 = math.isclose(actual_var_95, _EXPECTED_95_VAR, rel_tol=0.01)
        test_90 = math.isclose(actual_var_90, _EXPECTED_90_VAR, rel_tol=0.01)

        if not test_99:
            print(f"99% VAR test failed | Actual: {actual_var_99} | Expected: {_EXPECTED_99_VAR}")
        if not test_95:
            print(f"95% VAR test failed | Actual: {actual_var_95} | Expected: {_EXPECTED_95_VAR}")
        if not test_90:
            print(f"90% VAR test failed | Actual: {actual_var_90} | Expected: {_EXPECTED_90_VAR}")

        if all([test_99, test_95, test_90]):
            # All tests passed
            print("PortfolioVAR(): All VAR tests PASSED.\n")
            return True
        else:
            print("PortfolioVAR(): FAILED...\n")
            return False

    def ConditionalVAR(self, test_counter: int) -> int:
        """
        * ConditionalVAR()
        *
        * This function tests the conditional VAR calculation.
        * The function uses the test portfolio details and the 99%. 95% and 90% confidence
        * intervals to calculate the VAR. The results are compared to the expected values that
        * were calculated using the 'Portfolio_VAR.xlsx' file.
        """

        print(f"Running Test: ConditionalVAR() - Test {test_counter}...")

        confidence_1 = 0.99
        confidence_2 = 0.95
        confidence_3 = 0.90

        # NOTE: These numbers were caculated manually using the 'Portfolio_VAR.xlsx' file
        _EXPECTED_99_VAR = -0.02971
        _EXPECTED_95_VAR = -0.02032
        _EXPECTED_90_VAR = -0.01563

        # Calculate the conditional VAR at the confidence intervals
        actual_var_99 = self.stock_risk_engine.ConditionalVAR(confidence_interval=confidence_1)
        actual_var_95 = self.stock_risk_engine.ConditionalVAR(confidence_interval=confidence_2)
        actual_var_90 = self.stock_risk_engine.ConditionalVAR(confidence_interval=confidence_3)

        # Check actual vs expected (within +/- 5% for rounding errors)
        test_99 = math.isclose(actual_var_99, _EXPECTED_99_VAR, rel_tol=0.05)
        test_95 = math.isclose(actual_var_95, _EXPECTED_95_VAR, rel_tol=0.05)
        test_90 = math.isclose(actual_var_90, _EXPECTED_90_VAR, rel_tol=0.05)

        if not test_99:
            print(f"99% VAR test failed | Actual: {actual_var_99} | Expected: {_EXPECTED_99_VAR}")
        if not test_95:
            print(f"95% VAR test failed | Actual: {actual_var_95} | Expected: {_EXPECTED_95_VAR}")
        if not test_90:
            print(f"90% VAR test failed | Actual: {actual_var_90} | Expected: {_EXPECTED_90_VAR}")

        if all([test_99, test_95, test_90]):
            # All tests passed
            print("ConditionalVAR(): All VAR tests PASSED.\n")
            return True
        else:
            print("ConditionalVAR(): FAILED...\n")
            return False

    def MarginalLocalVAR(self, test_counter: int) -> int:
        """
        * MarginalLocalVAR()
        *
        * This function tests the marginal VAR calculation.
        * The function uses the test portfolio details and the 99%. 95% and 90% confidence
        * intervals to calculate the VAR. The results are compared to the expected values that
        * were calculated using the 'Portfolio_VAR.xlsx' file.
        """

        print(f"Running Test: MarginalLocalVAR() - Test {test_counter}...")

        confidence_intervals = [0.99, 0.95, 0.90]

        # NOTE: These numbers were caculated manually using the 'Portfolio_VAR.xlsx' file
        # _###_VAR[0] is at the 99% confidence interval
        # _###_VAR[1] is at the 95% confidence interval
        # _###_VAR[2] is at the 90% confidence interval
        _EXPECTED_VALUES = {
            "JPM" : [0.026402068, 0.018696743, 0.01450414],
            "NVDA" : [0.087757707, 0.062146016, 0.048210243],
            "LLY" : [0.022075387, 0.015632785, 0.012127]
        }

        all_tests = True

        for i in range(len(self.portfolio_details['Symbols'])):
            symbol = self.portfolio_details['Symbols'][i]
            ci = confidence_intervals[i]

            # Calculate the marginal VAR
            actual_marginal_var = self.stock_risk_engine.MarginalLocalVAR(symbol, ci)

            # Check actual vs expected (within +/- 1% for rounding errors)
            test = math.isclose(actual_marginal_var, _EXPECTED_VALUES[symbol][i], rel_tol=0.01)

            if not test:
                print(f"{ci*100}% Marginal VAR test failed | Actual: {actual_marginal_var} | Expected: {_EXPECTED_VALUES[symbol][i]}")
                all_tests = False

        if all_tests:
            print("MarginalLocalVAR(): All VAR tests PASSED.\n")
            return True
        else:
            print("MarginalLocalVAR(): FAILED...\n")
            return False

    def IncrementalLocalVAR(self, test_counter: int) -> int:
        """
        * IncrementalLocalVAR()
        *
        * This function tests the incremental VAR calculation.
        * The function uses the test portfolio details and the 99%. 95% and 90% confidence
        * intervals to calculate the VAR. The results are compared to the expected values that
        * were calculated using the 'Portfolio_VAR.xlsx' file.
        """

        print(f"Running Test: IncrementalLocalVAR() - Test {test_counter}...")

        confidence_intervals = [0.99, 0.95, 0.90]

        #                 JPM    NVDA   LLY
        weight_changes = [0.083, 0.167, -0.25]

        # NOTE: These numbers were caculated manually using the 'Portfolio_VAR.xlsx' file
        # _###_VAR[0] is at the 99% confidence interval
        # _###_VAR[1] is at the 95% confidence interval
        # _###_VAR[2] is at the 90% confidence interval
        _EXPECTED_VALUES = {
            "JPM" : [0.002191372, 0.00155183, 0.001203844],
            "NVDA" : [0.014655537, 0.010378385, 0.008051111],
            "LLY" : [-0.005518847, -0.003908196, -0.003031813]
        }

        all_tests = True

        for i in range(len(self.portfolio_details['Symbols'])):
            symbol = self.portfolio_details['Symbols'][i]
            ci = confidence_intervals[i]
            delta = weight_changes[i]

            # Calculate the incremental VAR
            actual_incremental_var = self.stock_risk_engine.IncrementalLocalVAR(symbol, delta, ci)

            # Check actual vs expected (within +/- 1% for rounding errors)
            test = math.isclose(actual_incremental_var, _EXPECTED_VALUES[symbol][i], rel_tol=0.01)

            if not test:
                print(f"{ci*100}% Incremental VAR test failed | Actual: {actual_incremental_var} | Expected: {_EXPECTED_VALUES[symbol][i]}")
                all_tests = False

        if all_tests:
            print("IncrementalLocalVAR(): All VAR tests PASSED.\n")
            return True
        else:
            print("IncrementalLocalVAR(): FAILED...\n")
            return False

    def ComponentLocalVAR(self, test_counter: int) -> int:
        """
        * ComponentLocalVAR()
        *
        * This function tests the component VAR calculation.
        * The function uses the test portfolio details and the 99%. 95% and 90% confidence
        * intervals to calculate the VAR. The results are compared to the expected values that
        * were calculated using the 'Portfolio_VAR.xlsx' file.
        """

        print(f"Running Test: ComponentLocalVAR() - Test {test_counter}...")

        confidence_intervals = [0.99, 0.95, 0.90]

        # NOTE: These numbers were caculated manually using the 'Portfolio_VAR.xlsx' file
        # _###_VAR[0] is at the 99% confidence interval
        # _###_VAR[1] is at the 95% confidence interval
        # _###_VAR[2] is at the 90% confidence interval
        _EXPECTED_VALUES = {
            "JPM" : [0.004409145, 0.003122356, 0.002422191],
            "NVDA" : [0.029223316, 0.020694623, 0.016054011],
            "LLY" : [0.011037694, 0.007816393, 0.006063626]
        }

        all_tests = True

        for i in range(len(self.portfolio_details['Symbols'])):
            symbol = self.portfolio_details['Symbols'][i]
            ci = confidence_intervals[i]

            # Calculate the incremental VAR
            actual_component_var = self.stock_risk_engine.ComponentLocalVAR(symbol, ci)

            # Check actual vs expected (within +/- 1% for rounding errors)
            test = math.isclose(actual_component_var, _EXPECTED_VALUES[symbol][i], rel_tol=0.01)

            if not test:
                print(f"{ci*100}% Component VAR test failed | Actual: {actual_component_var} | Expected: {_EXPECTED_VALUES[symbol][i]}")
                all_tests = False

        if all_tests:
            print("ComponentLocalVAR(): All VAR tests PASSED.\n")
            return True
        else:
            print("ComponentLocalVAR(): FAILED...\n")
            return False

    def DisplayPortfolioStatistics(self, test_counter: int) -> int:
        """
        * DisplayPortfolioStatistics()
        *
        * This function tests the display of the portfolio statistics. This test
        * always passes and is used to visually inspect the portfolio statistics.
        """

        print(f"Running Test: DisplayPortfolioStatistics() - Test {test_counter}...")

        # Display the portfolio statistics
        self.stock_risk_engine.DisplayPortfolioStatistics(plot=True)

        return True

    def run(self):
        """
        * run()
        *
        * Runs the sub-tests for the RiskEngine.
        """

        test_result = True
        test_counter = 1

        print('Running Unit Tests...')
        print('------------------------------------------')

        test_result |= self.Beta(test_counter)
        test_counter += 1

        test_result |= self.IndividualVAR(test_counter)
        test_counter += 1

        test_result |= self.PortfolioVAR(test_counter)
        test_counter += 1

        test_counter += self.ConditionalVAR(test_counter)
        test_counter += 1

        test_result |= self.MarginalLocalVAR(test_counter)
        test_counter += 1

        test_result |= self.IncrementalLocalVAR(test_counter)
        test_counter += 1

        test_result |= self.ComponentLocalVAR(test_counter)
        test_counter += 1

        test_result |= self.DisplayPortfolioStatistics(test_counter)
        test_counter += 1

        print('------------------------------------------')
        print('Unit Tests Complete. Results:')

        if test_result:
            print('ALL TESTS PASSED.')
        else:
            print(f"TESTS FAILED.")