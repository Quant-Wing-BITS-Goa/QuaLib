import numpy as np
import pandas as pd
from scipy.stats import norm
from math import sqrt, pi

class opt:
    """
    opt class:
    
    Args:
    df - dataframe conmprisinng of daily returns of each ticker

    Each different method returns a dictionary with weights assigned
    to the different tickers

    """

    def __init__(self, df):
        self.df = df

    def HRP():
        #code the Hierarchical Risk Parity Algorithm here 

    def markowitz(port):
        """
        The argument port can be sharpe, volatility in which cases
        you either maximize the sharpe ratio or minimize the volatility
        repectivle

        """
        if port=="sharpe":
            #code and return the weights for maximum sharpe
            def get_sr(weights):
                weights = np.array(weights)
                ret = np.sum(log_return.mean() * weights) * 252
                vol = np.sqrt(np.dot(weights.T, np.dot(log_return.cov() * 252, weights)))
                sr = ret/vol
                return sr
            def neg_sharpe(weights):
                 return  get_sr(weights) * -1
            def check_sum(weights):
                return np.sum(weights) - 1
            cons = ({'type':'eq','fun': check_sum})
            bounds = tuple((0,1) for x in range(len(df.columns)))
            init_guess=[]
            for x in range(len(df.columns)):
                init_guess.append(1/len(df.columns))
            log_return=np.log(df/df.shift(1))
            from scipy.optimize import minimize
            opt_sharp = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
            result=dict(zip(df.columns,opt_sharp .x))
            return result

        if port == "volatility":
            #code and return the weights for the portfolio with minimum volatility

    def original():
        #Leave this for now

