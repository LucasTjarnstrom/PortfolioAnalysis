from portfolio import Portfolio, Asset
from optimization import OptimizationEngine

# Create a portfolio with custom weights
# Print portfolio statistics and get various plots

p = Portfolio(name="Portfolio 1", valuation_date="2024-06-14", initial_investment=100)
p.add_holding("Avanza Zero", 0.5)
p.add_holding("Avanza Global", 0.5)
p.calculate_returns()
p.get_plots()
p.print_stats()

# Create an asset and print statistics and various plots

a = Asset(name="Länsförsäkringar Global Index", valuation_date="2024-06-14", initial_investment=100)
a.calculate_returns()
a.get_plots()
a.print_stats()

# Create a portfolio using the optimization engine

opt = OptimizationEngine.opt(name="Portfolio 2", valuation_date="2024-06-14", initial_investment=100)
opt.add_holding("Avanza Zero")
opt.add_holding("Avanza Global")
opt.add_holding("Länsförsäkringar Global Index")
opt.add_holding("Swedbank Robur Technology A")

# Hierarchical Risk Parity

opt.hierarchical_risk_parity(risk_param="CVaR")
opt.print_opt_stats()
opt.plot_optimization(include_bench=True)

# Mean-variance optimization

opt.mean_risk(opt_param="Maximize ratio", risk_param="Standard deviation")
opt.print_opt_stats()
opt.plot_optimization(include_bench=True)

# Inverse volatility weighting

opt.inverse_volatility()
opt.print_opt_stats()
opt.plot_optimization(include_bench=True)
