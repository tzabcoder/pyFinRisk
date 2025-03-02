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
            _prices.append(yf.download(symbol, start="2023-05-15", end="2024-05-15", auto_adjust=True)['Close'][symbol].to_list())

        self.portfolio_details["Prices"] = _prices

        # Download the market prices
        self.market_prices = yf.download('SPY', start="2023-05-15", end="2024-05-15", auto_adjust=True)['Close']['SPY'].to_list()

        # Initialize the risk engines
        self.stock_risk_engine = StockRiskEngine(self.portfolio_details, self.market_prices)

    def _basic_local_var(self):
        confidence = 0.99
        print(self.stock_risk_engine.BasicPortfolioVAR(confidence))

    def run(self):
        self._basic_local_var()