from RiskEngine import StockRiskEngine

holdings = {
    0.5: [0.05, 0.05, 0.05],
    0.25: [0.025, 0.025, 0.025],
    0.25: [0.01, 0.01, 0.01]
}

market_returns = [0.02, 0.02, 0.02]

risk_engine = StockRiskEngine(holdings)

beta = risk_engine.beta(market_returns)
print(beta)