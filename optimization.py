from portfolio import Portfolio, Asset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skfolio import RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.distance import PearsonDistance
from skfolio.optimization import EqualWeighted, MeanRisk, ObjectiveFunction, \
HierarchicalRiskParity, InverseVolatility
from skfolio.preprocessing import prices_to_returns
from skfolio.datasets import load_factors_dataset
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")





class OptimizationEngine(Portfolio):
    """ Inherits from Portfolio. Optimization engine for assets/portfolios using skfolio. """
    
    trading_days = 252
    test_size = 0.33
    
    def __init__(self, *portfolios: Portfolio):
        self._portfolios = [*portfolios]
        self._ranked_portfolios = pd.DataFrame()
        self._prices = pd.DataFrame()
        self._pred_model = str
        self._composition = []
    
    
    @classmethod
    def opt(cls, name: str, valuation_date: str, initial_investment: float):
        """ Instantiate an Asset object with an initial investment """
        
        obj = cls.__new__(cls)
        super(OptimizationEngine, obj).__init__(name, valuation_date, initial_investment)
        cls._portfolios = [obj]
        return obj
        
        
    @property
    def get_composition(self):
        """ Get optimized portfolio composition """
        return self._composition


    #@property
    #TODO: Rewrite
    def get_measure(self, measure):
        """ Get risk/optimization measure """
        
        if measure == "Maximize return":
            return ObjectiveFunction.MAXIMIZE_RETURN
        elif measure == "Maximize ratio":
            return ObjectiveFunction.MAXIMIZE_RATIO
        elif measure == "Minimize risk":
            return ObjectiveFunction.MINIMIZE_RISK
        elif measure == "CVaR":
            return RiskMeasure.CVAR
        elif measure == "Maximum drawdown":
            return RiskMeasure.MAX_DRAWDOWN
        elif measure == "Variance":
            return RiskMeasure.ANNUALIZED_VARIANCE
        elif measure == "Standard deviation":
            return RiskMeasure.ANNUALIZED_STANDARD_DEVIATION
        else:
            Exception("Measure {} not found".format(measure))
            
    
    def prepare_optimization(self):
        """ Get price data series """
        
        self._portfolios[0].set_prices()
        self._prices = self._portfolios[0].get_prices().set_index('date')
        
        returns = prices_to_returns(self._prices)
        train, test = train_test_split(returns, test_size=self.test_size, shuffle=False)
        return train, test
    
    
    def handle_output(self, model, train, test):
        """ Handle output from optimization models """
        
        model.fit(train)
        pred_model = model.predict(test)
        in_sample = [train.index[0].strftime('%Y-%m-%d'), train.index[-1].strftime('%Y-%m-%d')]
        out_of_sample = [test.index[0].strftime('%Y-%m-%d'), test.index[-1].strftime('%Y-%m-%d')]
        
        #model.hierarchical_clustering_estimator_.plot_dendrogram()
        
        self._pred_model = pred_model
        self._composition = pred_model.composition
        self._in_sample = in_sample
        self._out_of_sample = out_of_sample
        self._portfolio_returns = self._pred_model.returns_df
        self._portfolio_cum_returns = self._pred_model.cumulative_returns_df + 1
        self._portfolio_value = self._initial_investment*(self._pred_model.cumulative_returns_df.iloc[-1]+1)
        self.benchmark()
    
    
    def mean_risk(self, opt_param: str, risk_param: str):
        """ Mean Risk model """
        
        train, test = self.prepare_optimization()
        
        model = MeanRisk(
            risk_measure=self.get_measure(risk_param),
            objective_function=self.get_measure(opt_param),
            portfolio_params=dict(name="Minimum {} portfolio, {}".format(risk_param, opt_param)),
            )
        
        self._measure = opt_param
        self.handle_output(model, train, test)
        
        return model
    
    #TODO: Add features
    def hierarchical_risk_parity(self, risk_param: str):
        """ Hierarchical Risk Parity model """
        
        train, test = self.prepare_optimization()
        
        #self._portfolios[0].set_prices()
        #self._prices = self._portfolios[0].get_prices().set_index('date')
        #factors = load_factors_dataset()
        #returns, factor_returns = prices_to_returns(self._prices, factors[len(factors)-len(self._prices):])
        
        #train, test, factors_train, factors_test = train_test_split(returns, factor_returns, test_size=self.test_size, shuffle=False)
        
        model = HierarchicalRiskParity(
            risk_measure=self.get_measure(risk_param),
            distance_estimator=PearsonDistance(),
            portfolio_params=dict(name="Minimum {} portfolio".format(risk_param))
            )
        
        #model = HierarchicalRiskParity(risk_measure=RiskMeasure.CVAR,
        #                               hierarchical_clustering_estimator=HierarchicalClustering())
        self.handle_output(model, train, test)
        
        return model
    
    
    def inverse_volatility(self):
        """ Inverse Volatility model """
        
        train, test = self.prepare_optimization()
        
        model = InverseVolatility()
        
        self.handle_output(model, train, test)
        
        return model
    
        
    def benchmark(self):
        """ Benchmark portfolio, equally weighted. """
        
        X = prices_to_returns(self._prices)
        X_train, X_test = train_test_split(X, test_size=self.test_size, shuffle=False)
        benchmark = EqualWeighted(portfolio_params=dict(name="Equal Weighted"))
        benchmark.fit(X_train)
        bench_model = benchmark.predict(X_test)
        self._bench_model = bench_model
        self._benchmark_value = (self._bench_model.cumulative_returns_df.iloc[-1]+1)*self._initial_investment
        
    
    def plot_optimization(self, include_bench: bool = False):
        """ Get optimized plots """
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio balance")
        ax.set_title("Portfolio returns: {}".format(self._portfolios[0]._name))
        plt.plot(self._pred_model.cumulative_returns_df + 1, label=self._pred_model.name)
        if include_bench:
            plt.plot(self._bench_model.cumulative_returns_df + 1, label="Equally-Weighted")
        plt.legend(loc='upper left')
        
    
    def plot_benchmark(self):
        """ Get optimized plots """
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio balance")
        ax.set_title("Portfolio returns: {}".format(self._portfolios[0]._name))
        plt.plot(self._bench_model.cumulative_returns_df + 1, label="Equally-Weighted", color='b')
        plt.legend(loc='upper left')
     
        
    def plot_hrp(self):
        self._pred_model.plot_contribution(measure=RiskMeasure.CVAR)
        self._pred_model.hierarchical_clustering_estimator_.plot_dendrogram(heatmap=False)
        
        
    def get_plots(self):
        """ Get all plots """
        
        self.plot_benchmark()
        self.plot_optimization()
        
        
    def print_opt_stats(self) -> str:
        
        print("----- Valuation date: ", self._portfolios[0]._valuation_date," -----" \
              "\nIn-sample period: ", self._in_sample, \
              "\nOut-of-sample period: ", self._out_of_sample, \
              "\nHoldings and weights:\n" \
              "\n", self.get_composition, "\n" \
              "\n----- Optimization statistics -----" \
              "\nPortfolio value (benchmark): ", self._benchmark_value, \
              "\nPortfolio value (optimization): ", self._portfolio_value,)
        