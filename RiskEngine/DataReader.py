# Global Package Imports
import yfinance as yf
import pandas as pd

class DataReader:
    def __init__(self, file_path: str, xlxs: bool = True) -> None:
        """
        * __init__()
        *
        * Iniializes the DataReader class and sets the file path and file type.
        *
        * file_path: file to be read locally
        * xlxs: true if excel file, false if csv file
        """

        self._file_path = file_path
        self._xlsx = xlxs

    def ReadData(self) -> pd.DataFrame:
        """
        * ReadData()
        *
        * Reads the data from the file and in the mode specified in the constructor,
        * and returns a dataframe.
        *
        * :returns: dataframe containing the data read from the file
        """

        if self._xlsx:
            data = pd.read_excel(self._file_path)
        else:
            data = pd.read_csv(self._file_path)

        return data

    def MarketPrices(self, symbol: str, period: str, interval: str) -> list:
        """
        * MarketPrices()
        *
        * Downloads the market prices for the given symbol and returns a dataframe.
        *
        * symbol: market index symbol
        * period: period for the historical data to be downloaded
        * interval: interval for the historical data to be downloaded
        * :returns: list containing the market prices for the given symbol
        """

        market_prices = yf.download(symbol, period=period, interval=interval, auto_adjust=True)['Close'][symbol].tolist()
        return market_prices

    def CreateEquityPortfolioDetails(self, data: pd.DataFrame, symbols_id: str, qty_id: str, period: str, interval: str) -> dict:
        """
        * CreateEquityPortfolioDetails()
        *
        * Creates the portfolio dict for an equity-based portfolio. This function will also download the
        * historical data for the symbols in the portfolio.
        *
        * data: dataframe of the financial position data
        * symbols_id: column identifier for the portfolio symbols
        * qty_id: column identifier for the positon shares
        * period: period for the historical data to be downloaded
        * interval: interval for the historical data to be downloaded
        * :returns: dict containing the portfolio details
        """

        # Extract symbols and shares
        symbols = data[symbols_id].tolist()
        shares = data[qty_id].tolist()

        # Download the historical data and extract prices
        historical_data = yf.download(symbols, period=period, interval=interval, auto_adjust=True)['Close']
        prices = [historical_data[symbol].tolist() for symbol in symbols]

        portfolio_details = {
            "Symbols" : symbols,
            "Shares" : shares,
            "Prices" : prices
        }

        return portfolio_details