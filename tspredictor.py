import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels
from math import sqrt, pi

class tspredictor:
    """
    BSE Class
    
    Args:
    df: Dataframe consisting close price(time series data)

    """
    #Initialize a check to see if statsmodels is initialised. Feel free to use it any of the implementations below

    def __init__(self, df):
        self.df = df

    def AR(self,lags):

    def ARMA(self,order):
