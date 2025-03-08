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

* **Marginal VaR** - *Change* in Portfolio VaR resulting from taking adiitional (marginal) exposure to a given risk factor. Partial (linear) derivative with respect to the component's position.

$$
ΔVaR_i=\frac{δVaR}{δw_i}=α(β_i*σ_p)=\frac{VaR}{W}*β_i
$$

where $β_i$ is the component beta, and $W$ is the complete portfolio value (marked-to-market).

* **Incremental VaR** - *Change* in Portfolio VaR resulting from the midification of a portfolio component.

$$
IVaR_i = ΔVaR_i*a
$$

where a is a change, either positive or negative, in the component. It must be noted that this is only approximation.

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
* Marginal VaR
* Incremental VaR
* Component VaR
* First-Order Statistical Measures
* Metric Plotting

**Quick Start**

---

For a comprehensive demonstration of usage, view the .ipynb files in the examples folder. For test calculations, see the docs folder for the spreadsheet.

**Contributing**

---

If you would like to contribute to this project or notice any issues, please [raise an issue](https://github.com/tzabcoder/pyFinRisk/issues) or [create a pull request](https://github.com/tzabcoder/pyFinRisk/pulls).
