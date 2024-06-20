import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, t
from typing import Union, Dict, Literal, List
import database as db




class Portfolio:
    """ Parent class representing a portfolio object """
    
    trading_days = 252
    
    """ Create a portfolio by passing the name, valuation date and initial investment (optional) """
    def __init__(self, name: str, valuation_date: str, initial_investment: float = 1) -> None:
        
      self._name = name
      self._valuation_date = valuation_date
      self._holdings = dict() #Dict[str, float]
      self._weights = []
      self._cum_returns = pd.Series(dtype=float)
      self._returns = pd.DataFrame()
      self._portfolio_returns = pd.DataFrame()
      self._portfolio_cum_returns = pd.Series(dtype=float)
      self._portfolio_value = float
      self._simulation = pd.DataFrame()
      self._initial_investment = initial_investment
      self._corr = pd.DataFrame()
      self._ES = float
      self._h_VaR = float
      self._p_VaR = float
      self._dist = []
      self._kurtosis = float
      self._skew = float
      self._fee = float
      self._opt_data = pd.DataFrame()
      self._prices = pd.DataFrame()
      
      
    def add_holding(self, name: str, weight: float = 0) -> None:
        """ Add asset holdings and weight to portfolio """
        
        self._holdings.update({Asset(name, self._valuation_date):weight})
        self._weights.append(weight)
        
        
    @property
    def get_holdings(self) -> []:
        """ Get portfolio holdings by name """ 
        
        names = []
        for fund in self._holdings.keys():
            names.append(fund._name)
        return names
    
    
    @property
    def get_value(self) -> float:
        """ Get portfolio value """
        
        if isinstance(self, Asset):
            return self._value
        else:
            return self._portfolio_value
    
    
    @property
    def get_opt_data(self) -> pd.DataFrame:
        """ Get data required to optimize """
        return self._opt_data
    
    
    @property
    def get_weights(self) -> []:
        """ Get portfolio weights """
        return self._weights
    
    
    @property
    def get_valuation_date(self) -> str:
        """ Get valuation date """
        return self._valuation_date
    
    
    #@property
    def get_prices(self) -> pd.DataFrame:
        """ Get portfolio prices """
        return self._prices
    
    
    #@get_prices.setter
    def set_prices(self) -> None:
        """ Set prices """
        
        data_frame = []
        for fund in self._holdings.keys():
            data_frame.append(fund.get_data())
        
        prices = pd.concat(data_frame, axis=1).reset_index()
        prices = prices[prices['date'] <= self._valuation_date].dropna()
        self._prices = prices
    
    
    def set_weights(self) -> None:
        """ Set asset weight """
        pass
        
    
    def set_opt_data(self) -> None:
        """ Set DataFrame required to optimize """
        
        self._opt_data = pd.DataFrame(data=[self._valuation_date, self.cagr(), self.volatility(), self.sharpe_ratio(),
                             self.max_draw_down(), self.cvar(), self.kurtosis(), self.skew()],
                            index=["date", "cagr", "volatility", "sharpe", "drawdown",
                                   "cvar", "kurtosis", "skew"],
                            columns=[self._name])
    
    
    def portfolio_returns(self, returns: pd.DataFrame) -> None:
        """ Concatenate portfolio holdings and calculate returns """
        
        if abs(np.sum(self._weights) - 1) > 1e5:
            raise Exception("The weights must sum to 1 in order to calculate portfolio returns.")
        
        #self.calculate_returns()
        
        if len(returns.columns) > 1:
            weights = np.array(self._weights) 
            portfolio_returns = returns.dot(weights)
        
        self._portfolio_returns = portfolio_returns
        self._portfolio_cum_returns = (portfolio_returns+1).cumprod()
        self._portfolio_value = self._portfolio_cum_returns.iloc[-1]*self._initial_investment
        self.set_opt_data()


    def calculate_returns(self) -> pd.DataFrame:
        """ Calculate returns for a fund or portfolio """
        
        if isinstance(self, Asset):
            returns = self.get_returns()
            self._value = self._cum_returns.iloc[-1]*self._initial_investment
        else: 
            data_frame = []
            for fund in self._holdings.keys():
                data_frame.append(fund.get_returns())
            returns = pd.concat(data_frame, axis=1).dropna()
            self._returns = returns
            self._cum_returns = (returns+1).cumprod()
            self.portfolio_returns(returns)

        return returns
    
    
    def rebalance(self, period: str = '1M') -> pd.DataFrame:
        """ Rebalance every interval given by period """
        
        returns = self._portfolio_returns
        
        # .sum() summerar returns för varje period. Bör vara log returns.
        returns_sampled = returns.resample(period, convention='end').sum()
        weights = np.array(self._weights)
        returns_sampled = returns_sampled.dot(weights) #.sum(axis=1) 
        returns = returns_sampled

        return returns
    
    
    def draw_down(self) -> pd.Series:
        """ Draw down """
        
        if isinstance(self, Asset):
            cumulative_returns = self._cum_returns
        else:
            cumulative_returns = self._portfolio_cum_returns
            
        running_max = np.maximum.accumulate(cumulative_returns)
        return ((cumulative_returns-running_max)/running_max)
    
    
    def max_draw_down(self) -> float:
        """ Max draw down """
        return self.draw_down().min()
    
    
    def value_at_risk(self, dist: str = 'normal', alpha: float = 0.95, dof: int = 6) -> float:
        """ Annualized VaR """
        
        if isinstance(self, Asset):
            mean = self._returns.mean()
            std = self._returns.std()
        else:
            mean = self._portfolio_returns.mean()
            std = self._portfolio_returns.std()
        
        if dist == 'normal':
            var = norm.ppf((1-alpha), mean, std)
        elif dist == 'student-t':
            nu = dof
            var = np.sqrt((nu-2)/nu) * t.ppf(1-alpha, nu) * std - mean
     
        return var
    
    
    def historical_var(self, alpha: float = 0.95) -> float:
        """ Historical VaR """
        
        if isinstance(self, Asset):
            returns_sorted = self._returns.sort_values(ascending=True).dropna().reset_index(drop=True)
        else:
            returns_sorted = self._portfolio_returns.sort_values(ascending=True).dropna().reset_index(drop=True)
        
        #sorted_returns = sorted_returns - sorted_returns.mean()
        return np.percentile(returns_sorted, 100 * (1-alpha))
    
    
    def cvar(self, dist: Literal['normal', 'student-t'] = 'normal', alpha: float = 0.95, dof: int = 6) -> float:
        """ Annualized CVaR """
        
        if isinstance(self, Asset):
            mean = self._returns.mean()
            std = self._returns.std()
        else:
            mean = self._portfolio_returns.mean()
            std = self._portfolio_returns.std()
        
        if dist == 'normal':
            cvar = (1-alpha)**-1 * norm.pdf(norm.ppf(1-alpha)) * std - mean
        elif dist == 'student-t':
            nu = dof
            xanu = t.ppf(1-alpha, nu)
            cvar = 1/(1-alpha) * (1-nu)**(-1) * (nu-2+xanu**2) * t.pdf(xanu, nu) * std - mean
            
        return cvar
    
    
    def historical_cvar(self, alpha: float = 0.95) -> float:
        """ Historical CVaR """
        
        if isinstance(self, Asset):
            returns = self._returns
        else:
            returns = self._portfolio_returns
        
        var = self.value_at_risk(alpha)
        cvar = np.nanmean(returns[returns < var])
        return cvar
    
    
    def correlation(self):
        """ Linear correlation """
        
        returns = self._returns
        
        if isinstance(self, Asset):
            correlation = 1
        else:
            correlation = returns.corr(method='pearson')
            
        self._corr = correlation
        return correlation
    
    
    def volatility(self) -> float:
        """ Annualized volatility """
        
        if isinstance(self, Asset):
            volatility = self._returns.std()*np.sqrt(self.trading_days)
        else: 
            volatility = self._portfolio_returns.std()*np.sqrt(self.trading_days)
        
        return volatility
    
    
    def cagr(self) -> float:
        """ Cumulative annual growth return """
        
        if isinstance(self, Asset):
            cumulative_returns = self._cum_returns
        else:
            cumulative_returns = self._portfolio_cum_returns
        
        num_days = len(cumulative_returns)
        return (np.power(cumulative_returns.iloc[-1], 1/((num_days/self.trading_days)))-1)
    
    
    def gross_return(self) -> float:
        """ Gross (total) return of the portfolio """
        
        if isinstance(self, Asset):
            cumulative_returns = self._cum_returns
        else:
            cumulative_returns = self._portfolio_cum_returns
            
        return (cumulative_returns.iloc[-1]-1)
    
    
    def sharpe_ratio(self, r_f: float = 0) -> float:
        """ Sharpe ratio """
        
        sigma = self.volatility()
        r = self.cagr()
        return (r-r_f)/sigma
    
    
    def sortino_ratio(self, r_f: float = 0) -> float:
        """ Sortino ratio: sharpe ratio of downside deviation """
        
        if isinstance(self, Asset):
            returns = self._returns
        else:
            returns = self._portfolio_returns
        
        returns_risk_adj = np.asanyarray(returns - r_f)
        mean_annual_return = returns_risk_adj.mean() * self.trading_days
        downside_diff = np.clip(returns_risk_adj, np.NINF, 0)
        np.square(downside_diff, out=downside_diff)
        annualized_downside_deviation = np.sqrt(downside_diff.mean()) * np.sqrt(self.trading_days)
    
        return mean_annual_return / annualized_downside_deviation
    
    
    def omega_ratio(self, required_return: float = 0) -> float:
        """ The omega ratio is the quotient between the weighted positive and negative returns."""
        
        if isinstance(self, Asset):
            returns = self._returns
        else:
            returns = self._portfolio_returns
        
        return_threshold = (1 + required_return) ** (1 / self.trading_days) - 1

        returns_less_thresh = returns - return_threshold

        numer = np.sum(returns_less_thresh[returns_less_thresh > 0.0])
        denom = -1.0 * np.sum(returns_less_thresh[returns_less_thresh < 0.0])

        if denom > 0.0:
            return numer / denom
        else:
            return np.nan
        

    def skew(self) -> float:
        """ Portfolio skew """
        
        if isinstance(self, Asset):
            return self._returns.skew()
        else:
            return self._portfolio_returns.skew()
    
    
    def kurtosis(self) -> float:
        """ Portfolio kurtosis """
        
        if isinstance(self, Asset):
            return self._returns.kurtosis()
        else:
            return self._portfolio_returns.kurtosis()
    
    
    def weighted_fee(self) -> float:
        """ Weighted fee of portfolio """
        
        weighted_fee = []
        for fund, weights in self._holdings.items():
            weighted_fee.append(fund._fee * weights)

        return np.sum(weighted_fee)
    
        
    def print_stats(self, plots: bool = False) -> Union[str, plt]:
        """ Print portfolio statistics """
        
        #returns = np.log(1+self._returns).sum()+1
        if isinstance(self, Asset):
            fee = round(self._fee, 2)
            holdings = self._name
            weights = 1
        elif isinstance(self, Portfolio):
            fee = round(self.weighted_fee(), 2)
            holdings = self.get_holdings
            weights = self._weights
        
        initial_investment = self._initial_investment
        value = self.get_value
        total_return = round(self.gross_return()*100,2)
        annualized_return = round(self.cagr()*100,2)
        std_dev = round(self.volatility()*100,2)
        sharpe_ratio = round(self.sharpe_ratio(), 2)
        sortino_ratio = round(self.sortino_ratio(), 2)
        omega_ratio = round(self.omega_ratio(), 2)
        kurtosis = round(self.kurtosis(), 2)
        skew = round(self.skew(), 2)
        hist_var = round(self.historical_var()*100, 2)
        param_var = round(self.value_at_risk()*100, 2)
        param_var_t = round(self.value_at_risk('student-t')*100, 2)
        hist_cvar = round(self.historical_cvar()*100, 2)
        param_cvar = round(self.cvar()*100, 2)
        param_cvar_t = round(self.cvar('student-t')*100, 2)
        max_draw = round(self.max_draw_down()*100, 2)
        print("----- Valuation date: ", self._valuation_date," -----" \
              "\nHoldings: ", holdings, \
              "\nWeights: ", weights, \
              "\n----- Portfolio statistics: ", self._name," -----" \
              "\nInitial investment (SEK): ", initial_investment, \
              "\nPortfolio value (SEK): ", value, \
              "\nTotal return: ", total_return, "%" \
              "\nAnnualized return: ", annualized_return, "%" \
              "\nWeighted fee: ", fee, "%" \
              "\nSharpe ratio: " , sharpe_ratio, \
              "\nSortino ratio: " , sortino_ratio, \
              "\nOmega ratio: " , omega_ratio, \
              "\n----- Risk metrics -----" \
              "\nStd. deviation: ", std_dev, "%" \
              "\nKurtosis: ", kurtosis, \
              "\nSkew: ", skew, \
              "\n95% historical VaR: ", hist_var, "%" \
              "\n95% parametric VaR (normal): ", param_var, "%" \
              "\n95% parametric VaR (student-t): ", param_var_t, "%" \
              "\n95% historical CVaR: ", hist_cvar, "%" \
              "\n95% CVaR (normal): ", param_cvar, "%" \
              "\n95% CVaR (student-t): ", param_cvar_t, "%" \
              "\nMaximum drawdown: ", max_draw, "%" \
              "\nCorrelation matrix: \n", self.correlation())
        if plots:
            self.get_plots()
    
    
    def plot_returns(self) -> None:
        """ Get cumulative returns plot """
        
        if isinstance(self, Asset):
            returns = self._cum_returns
        else:
            returns = self._portfolio_cum_returns
        
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlabel("Date")
        ax.set_ylabel("Balance")
        ax.set_title("Returns: {}".format(self._name))
        plt.plot(returns, label=self._name, color='b')
        plt.legend(loc='upper left')
   
            
    def plot_drawdown(self) -> None:
        """ Get drawdown plot """

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns")
        self.draw_down().plot(title="Draw down: {}".format(self._name), kind="area", color="salmon", alpha=0.5)


    def plot_assets(self) -> None:
        """ Get portfolio returns by asset plot """
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative returns")
        ax.set_title("Individual asset returns for: {}".format(self._name))
        plt.plot(self._cum_returns, label=self.get_holdings)
        plt.legend(loc='upper left')
        
        
    def plot_distribution(self) -> None:
        """ Get portfolio returns by asset plot """
        
        if isinstance(self, Asset):
            returns = self._returns
        elif isinstance(self, Portfolio):
            returns = self._portfolio_returns
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns")
        ax.set_title("Return distribution: {}".format(self._name))
        plt.hist(returns, bins=100, density=True, label=self._name)
        plt.legend(loc='upper left')
        
        
    def get_plots(self) -> None:
        """ Get all plots """
        
        if not isinstance(self, Asset):
            self.plot_assets()
        
        self.plot_returns()
        self.plot_distribution()
        #self.plot_assets()
        self.plot_drawdown()
        
        
        


class Asset(Portfolio):
    """ Inherits from Portfolio. Represents a generic asset. """
    
    def __init__(self, name: str = "", valuation_date: str = "", initial_investment: float = 1):
        self._name = name
        self._valuation_date = valuation_date
        self._value = float
        self._initial_investment = initial_investment
        self._fee = float
        self._returns = pd.DataFrame() 
        self._cum_returns = pd.Series(dtype=float)
        self._prices = pd.DataFrame()
        self._simulation = pd.DataFrame()
        
    
    def get_returns(self, log_return: bool = False) -> pd.Series:
        """ Calculate asset returns """
        
        if self._returns.empty:
            self.get_data()
        
        if log_return:
            returns = np.log(1+self._prices.pct_change).dropna()
        else:
            returns = self._prices.pct_change().dropna()
        
        returns = returns[returns.index <= self._valuation_date].squeeze()
        self._returns = returns
        self._cum_returns = (returns+1).cumprod()
        return returns
        
        
    def get_data(self) -> pd.DataFrame:
        """ Get historical data for an asset. Update db if there's new data available. """
        
        data = db.get_data_range(self._name, db.C_DB)
        #data = data.rename(columns={'price':self._name})
        #print(data)
        
        # TODO: date > self._valuation_date
        #if data.empty or (data.date.iloc[-1].split(sep=' ')[0] != self._valuation_date):
        if data.empty or (data.date.iloc[-1] != self._valuation_date):
            try: 
                self.update_data()
                data = db.get_data_range(self._name, db.C_DB)
            except: 
                print("The asset can not be found.")
            if data.empty:
                raise Exception("The asset {} can not be found.".format(self._name))

        self._prices = data[['date', 'price']].set_index('date').rename(columns={'price':self._name})
        self._fee = data.fee.iloc[0]
        return self._prices
     
        
    def check_date(data, self) -> bool:
        """ Check if data needs to be updated """

        if data.empty:
            return True
        else:
            data.index[-1].split(sep=' ')[0] == self._valuation_date
            return False
            
        
    def update_data(self) -> None:
        """ Access database and save data """
        db.save_fund_data(self._name, db.C_DB)
