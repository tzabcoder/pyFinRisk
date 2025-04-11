from RiskEngine.DataReader import DataReader
from RiskEngine.StockRiskEngine import StockRiskEngine

FILE_PATH = 'E:/python_packages/pyFinRisk/examples/data/ic_positions.xlsx'

# Read the portfolio data
reader = DataReader(FILE_PATH, xlxs=True)
data = reader.ReadData()

# Create the details of the portfolio
details = reader.CreateEquityPortfolioDetails(
    data,
    symbols_id='Symbol',
    qty_id='Shares',
    period='2y',
    interval='1d'
)

# Download the market data
market_prices = reader.MarketPrices(symbol='SPY', period='2y', interval='1d')

# Create the risk engine
engine = StockRiskEngine(details, market_prices)

# Display the portfolio statistics
statistics = engine.PortfolioStatistics(display=True, plot=True)

# Calculate the Portfolio VaR (% and $) ==========
# Portfolio VaR is calculated at the 99% confidence level
dollar_pVar = engine.PortfolioVAR(confidence_interval=0.01, dollar_based=True)
percent_pVar = engine.PortfolioVAR(confidence_interval=0.01)

print(f"Historical Portfolio Value at Risk: ${dollar_pVar}")
print(f"Historical Portfolio Value at Risk:  {percent_pVar*100}%")

# Simulate the portfolio VaR at the 99% confidence level
simulated_var = engine.SimulatedPortfolioVAR(sims=1_000, n=20, confidence_interval=0.01)

print(f"Simulated Portfolio Value at Risk: ${simulated_var}")

# Calculate Positional VaR Components (%) ==========
symbols = data['Symbol'].to_list()

for symbol in symbols:
    marginal_var = engine.MarginalLocalVAR(symbol=symbol, confidence_interval=0.01)
    incremental_var = engine.IncrementalLocalVAR(symbol=symbol, weight_change=0.05, confidence_interval=0.01,)
    component_var = engine.ComponentLocalVAR(symbol=symbol, confidence_interval=0.01)

    print(f"{symbol} (99%): Marginal={round((marginal_var*100),4)}% | Incremental={round((incremental_var*100),4)}% | Component={round((component_var*100),4)}%")