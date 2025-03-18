# Local Imports
from RiskEngine.DataReader import DataReader
from RiskEngine.StockRiskEngine import StockRiskEngine

def main():
    # Create the data reader
    # NOTE: This file and directory must exist for the code to run
    data_reader = DataReader('data/Account_Positions.xlsx', xlxs=True)
    data = data_reader.ReadData()
    portfolio_details = data_reader.CreateEquityPortfolioDetails(
        data,
        symbols_id = 'Symbol',
        qty_id = 'Qty',
        period = '5y',
        interval = '1d'
    )
    market_prices = data_reader.MarketPrices('SPY', '5y', '1d')

    # Create the risk engine
    risk_engine = StockRiskEngine(portfolio_details, market_prices)

    # Calculate risk metrics =================================================
    risk_engine.DisplayPortfolioStatistics(True)

    # Calculate individual VaR
    for symbol in portfolio_details['Symbols']:
        print(f"{symbol} Individual VaR: {round(risk_engine.IndividualVAR(symbol, confidence_interval=0.01)*100, 4)}%")

    print(f"Portfolio VaR:   {round(risk_engine.PortfolioVAR(confidence_interval=0.01)*100, 4)}%")
    print(f"Conditional VaR: {round(risk_engine.PortfolioVAR(confidence_interval=0.01)*100, 4)}%")
    # ========================================================================

if __name__ == '__main__':
    main()