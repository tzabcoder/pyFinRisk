# pyFinRisk

pyFinRisk (Python Financial Risk) is a Python package for quantifying and measuring financial risk assumed by a portfolio of assets. The risk tools focus heavily on Value at Risk (VaR).

**Overview**

---

Value at Risk (VaR) is a method of assessing financial risk using standard statistical techniques summarizing the worst-loss over a target horizon that will not be exceeded with a given confidence interval.

Ex: Take the 99% confidence interval. VaR is the cutoff such that the probability of experiencing a greater loss than X is less than 1%.

* **Portfolio VaR** - Portfolio VaR measures the total value at risk of a portfolio of components (risk factors).

$$
σ_p^2 = \begin{bmatrix} w_1 & w_2 & ... & w_n \end{bmatrix}\begin{bmatrix} σ_1^2 & σ_{12} & ... & σ_{1N} \\ ... & ... & ... & ... \\ σ_{N1} & σ_{N2} & ... & σ_N^2 \end{bmatrix}\begin{bmatrix} w_1 \\ w_2 \\ ... \\ w_n \end{bmatrix}
$$

$$
σ_p^2 = w'Σw
$$

where, Σ is the covariance matrix and w is a vector of weights.

$$
Portfolio\hspace{0.1cm}VaR = ασ_pw = α\sqrt{w'Σw}
$$

* **Conditional VaR** - Value at Risk measures the estimated average loss in extreme scenarios beyond the VaR limit.

$$
CVaR_α=E[X|X<=VaR_α]
$$

* **Marginal VaR** - *Change* in Portfolio VaR resulting from taking additional (marginal) exposure to a given risk factor. Partial (linear) derivative with respect to the component's position.

$$
ΔVaR_i=\frac{δVaR}{δw_i}=α(β_i*σ_p)=\frac{VaR}{W}*β_i
$$

where $β_i$ is the component beta, and $W$ is the complete portfolio value (marked-to-market).

* **Incremental VaR** - *Change* in Portfolio VaR resulting from the modification of a portfolio component.

$$
IVaR_i = ΔVaR_i*a
$$

where a is a change, either positive or negative, in the component. It must be noted that this is only an approximation.

* **Component VaR** - *Change* in Portfolio VaR resulting from the removal of a risk factor.

$$
CVaR_i=(ΔVaR_i)*w_iW=\frac{VaR*β_i}{W}*w_iW=VaR*β_i*w_i
$$

where $w_i$ is the weight of the risk factor and $β_i$ is the component beta. It must be noted that this is only an approximation and works well with larger portfolios.

**Features**

---

For equity-based portfolios:

* Beta
* Individual (positional) VaR
* Portfolio VaR
* Conditional VaR
* Marginal VaR
* Incremental VaR
* Component VaR
* First-Order Statistical Measures
* Metric Plotting

**Quick Start**

---

For a comprehensive demonstration of usage, view the .py files in the examples folder. For test calculations, see the docs folder for the spreadsheet.

```python
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
market_prices = yf.download('SPY', period='1y', interval='1d', auto_adjust=False)['Close'][ticker].to_list()

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
```

**Testing**

---

For running the unit tests, execute the following command from the top-level directory:

`python -m test.test`

For running the example, execute the following command from the top-level directory:

`python -m examples.stock_risk_ex`

**Contributing**

---

If you would like to contribute to this project or notice any issues, please [raise an issue](https://github.com/tzabcoder/pyFinRisk/issues) or [create a pull request](https://github.com/tzabcoder/pyFinRisk/pulls).
