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

        if port == "volatility":
            #code and return the weights for the portfolio with minimum volatility

    def original():
        #Leave this for now

