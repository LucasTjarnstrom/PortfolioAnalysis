from portfolio import Portfolio, Asset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.copula.api import CopulaDistribution, GumbelCopula, IndependenceCopula, \
    GaussianCopula
from scipy.stats import norm
from arch import arch_model





class RiskEngine(Portfolio):
    """ Inherits from Portfolio. Risk Engine to compute risk calculations. """
    
    trading_days = 252
    
    def __init__(self, *portfolios: Portfolio):
        self._portfolios = [*portfolios]
        self._simulation = pd.DataFrame()
        self._ranked_portfolios = pd.DataFrame()
        self._garch_vol = pd.DataFrame()
    
    
    def get_garch_vol(self):
        """ Get the GARCH volatility """
        return self._garch_vol.iloc[-1]
    
    
    def prepare_sim(self):
        """ Load required data from the portfolio instance """
        
        for i in range(len(self._portfolios)):
            self._portfolios[i].calculate_returns()
        self._portfolio_returns = self._portfolios[i]._portfolio_returns
        self._portfolio_cum_returns = self._portfolios[i]._portfolio_cum_returns
        self._portfolio_value = self._portfolios[i]._portfolio_value
        self._returns = self._portfolios[i]._returns
        self._name = self._portfolios[i]._name
        self._holdings = self._portfolios[i].get_holdings
        self._weights = self._portfolios[i].get_weights
    
    
    def garch(self):
        """ Fit a GARCH model """
        
        # Rescale returns for better convergence
        g = arch_model(self._portfolio_returns*100, vol='GARCH', mean='Constant', p=1, q=1, dist='Normal')
        model = g.fit()
        self._garch_vol = model.conditional_volatility*np.sqrt(self.trading_days)/100
        return model
    
    #TODO: Not done here
    def forecast_vol(self):
        """ Forecast volatility using GARCH """
        
        model = self.garch()
        
        # Forecast vol the next 5 days
        model_forecast = model.forecast(horizon=60)

        # Plot forecasted volatility
        self._garch_forecast = pd.DataFrame(np.sqrt(model_forecast.variance.dropna().T * self.trading_days))
        #plt.plot(volatility, title="Volatility forecast using GARCH")
        #fdf.plot()
        #fdf.columns = ['Cond_Vol']
        #print(fdf['Cond_Vol'])
        #plt.plot(fdf['Cond_Vol'])#, label="Horizon", title='GARCH Volatility Forecast')
        #plt.label(loc='upper left')
    
    
    def gaussian_copula(self):
        """ Gaussian Copula """
        
        copula = GaussianCopula(corr=self.correlation())
        copula.plot_pdf()
        return copula
    
    
    #TODO: Not done here
    def forecast_var(self, alpha: float = 0.95):
        """ Forecast VaR using a copula """
        
        copula = self.gaussian_copula()
        # Simulate rvs from the copula dependency structure
        obs = copula.rvs(1000)
        # Transform into a distribution
        unif = norm.cdf(obs)
        # Get the marginal distribution
        marginal = norm.ppf(unif)
        # Compute VaR
        var = np.quantile(marginal, 1-alpha)*100
        return var
    
    
    def mc_var(self, alpha: float = 0.95):
        """ Monte-Carlo VaR """
        
        simulation = self._simulation
        sorted_trajectories = simulation.sort_values(by=self._num_trajectories-1).reset_index()
        var_path = sorted_trajectories[self._num_trajectories-1].sort_values(ascending=True)
        var = np.quantile(var_path, 1-alpha)
        var_pct = ((self._portfolio_value-var)/self._portfolio_value)*100
        return var_pct
        

    def wiener_process(self, delta: float, sigma: float, time_steps: float, paths: float) -> np.array:
        """ Simulate a Wiener process """

        # Returns an array of samples from a normal distribution
        return sigma * np.random.normal(loc=0, scale=np.sqrt(delta), size=(time_steps, paths))


    def gbm(self, delta: float, sigma: float, time_steps: float, mu: float, paths: float) -> np.array:
        """ Simulate Geometric Brownian Motion """
        
        process = self.wiener_process(delta, sigma, time_steps, paths)
        return np.exp(process + (mu - sigma**2 / 2) * delta)
    

    def sim(self, time_steps: float, paths: float, delta: float = 1/trading_days, garch: bool = False) -> pd.DataFrame:
        """ Returns price paths for the GBM simulation """
        
        if garch:
            volatility = self.get_garch_vol()
        else:
            volatility = self.volatility()
        
        value = self._portfolio_value
        
        returns = self.gbm(delta, volatility, time_steps, self.cagr(), paths)
        stacked = np.vstack([np.ones(paths), returns])
        self._simulation = pd.DataFrame(value * stacked.cumprod(axis=0))
        self._num_trajectories = paths
    
    
    def plot_simulation(self):
        """ Plot simulations"""
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlabel("Days")
        ax.set_ylabel("Portfolio balance")
        ax.set_title("Portfolio returns: {}".format(self._name))
        plt.plot(self._simulation, color='b')
        
        
    def plot_garch(self):
        """ Plot conditional volatility estimated by the GARCH model """
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.set_title("Conditional historical volatility (GARCH): {}".format(self._name))
        plt.plot(self._garch_vol, color='b')
        
        
    def plot_garch_forecast(self):
        """ Plot the volatility forecast estimated by the GARCH model """
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.set_title("Volatility forecast (GARCH): {}".format(self._name))
        plt.plot(self._garch_forecast, color='b')
        
        
    def print_risk_stats(self) -> str:
        """ Print risk stats """
        
        print("----- Valuation date: ", self._portfolios[0]._valuation_date," -----" \
              "\nHoldings: ", self._holdings, \
              "\nWeight: ", self._weights, \
              "\nPortfolio value: ", self._portfolio_value, \
              "\n----- Risk statistics -----" \
              "\nGARCH volatility: ", self.get_garch_vol(), "%"  \
              "\n95% Monte-Carlo VaR: ", self.mc_var(), "%"  \
              "\n----- Forecast -----" \
              "\nGARCH volatility 1-day forecast: ", self._garch_forecast.iloc[-1].values[0], "%"  \
              "\n95% VaR: ", self.forecast_var(), "%" )
        
        
        
if __name__ == "__main__":
    
    p1 = Portfolio("Portfolio 1", "2024-03-25", 100)
    p1.add_holding("Avanza Zero", 0.5)
    p1.add_holding("Avanza Global", 0.5)
    
    r = RiskEngine(p1)
    r.prepare_sim()
    r.forecast_vol()
    r.sim(100, 100, garch=True)
    r.plot_simulation()
    r.plot_garch()
    r.forecast_var()
    r.plot_garch_forecast()
    r.print_risk_stats()