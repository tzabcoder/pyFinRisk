# Global Imports
import yfinance as yf

# Local Imports
from RiskEngine.StockRiskEngine import StockRiskEngine

# Create the portfolio details (each ticker index corresponds with
# the shares index)
tickers = [
    'AAPL','NVDA','GOOG','UNH','LLY',
    'GAP','HD','COST','KO','AMGN',
    'VZ','BMY','META','DIS','JPM',
    'MA','NKE','PG','WMT','PM'
]

shares = [
    40, 40, 40, 40, 40,
    30, 30, 30, 30, 30,
    20, 20, 20, 20, 20,
    10, 10, 10, 10, 10
]

prices = []
for ticker in tickers:
    # Download the past 1 years worth of data
    prices.append(yf.download(ticker, period='1y', interval='1d', auto_adjust=False)['Close'][ticker].to_list())

portfolio_details = {
    'Symbols' : tickers,
    'Shares' : shares,
    'Prices' : prices
}

# Download the market prices
market_prices = yf.download('SPY', period='1y', interval='1d', auto_adjust=False)['Close']['SPY'].to_list()

# Create the risk engine
riskEngine = StockRiskEngine(portfolio_details, market_prices)

# Display and plot the portfolio statistics
riskEngine.DisplayPortfolioStatistics(plot=True)

# Calculate the Individual VaR for a component
individual_var = riskEngine.IndividualVAR(symbol='JPM', confidence_interval=0.95)

# Calculate the Portfolio VaR
portfolio_var_pct = riskEngine.PortfolioVAR(confidence_interval=0.99) # Returns %
portfolio_var_dollar = riskEngine.PortfolioVAR(confidence_interval=0.99, dollar_based=True) # Returns $

# Calculate the Marginal VaR
marginal_var = riskEngine.MarginalLocalVAR(symbol='AAPL')

# Calculate the Incremental VaR
incremental_var = riskEngine.IncrementalLocalVAR(symbol='COST', weight_change=0.05)

# Calculate the Component VaR
component_var = riskEngine.ComponentLocalVAR(symbol='LLY')