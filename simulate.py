from scipy.stats import norm
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


""""
This module will help you
to simulate the movement of stock pricing
using different discretization techniques

Generally all the functions have 3 arguments:
S0: Initial Stock Price
T: how many days the simulation has to predict
n: no. of different MC simultions

The different functions sould return a nXT matrix


plot_sim allows you to print upto 10 simulations of the mentioned function in format (s0,T,n,func)
"""

def asset(S0,T,n):
    returns =data.pct_change()
    u = returns.mean()
    var = returns.var()

    drift =  np.array(u - (0.5*var))  #drift  term
    stdev = returns.std() # volatility term

    a1 = stdev*norm.ppf(np.random.rand(T,n))

    price_list = np.zeros(shape=(T,n), dtype=float, order='F')

    price_list[0] = S0
    for t in range(1,T):
        price_list[t] = price_list[t-1]*((a1[t]*(t**0.5)) + (drift*t) + 1)


        return price_list # Tx n matrix
    
def currency(S0,T,n):
    rf = 0.02 # risk free rate
    returns =data.pct_change()
    drift =  returns.mean() -rf #drift  term
    stdev = returns.std() # volatility term

    a1 = stdev*norm.ppf(np.random.rand(T,n))

    price_list = np.zeros(shape=(T,n), dtype=float, order='F')

    price_list[0] = S0
    for t in range(1,T):
        price_list[t] = price_list[t-1]*((a1[t]*(t**0.5)) + (drift*t) + 1)

    return price_list # Tx n matrix
def coxwell(S0,T,n):
    
def expBM(S0,T,n):
    
    # defining required variables
    log_returns = np.log(1 + data.pct_change())
    u = log_returns.mean()
    stdev = log_returns.std()
    var = log_returns.var()

    drift =  np.array(u - (0.5*var))
    
    # generating a random matrix based on previos data

    daily_returns = np.exp(stdev*norm.ppf(np.random.rand(T,n)) + drift)

    # apply it to generate different price lists
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0
    for t in range(1,T):
        price_list[t] = price_list[t-1]*daily_returns[t]

    #      plt.plot(price_list)
    return price_list #output nXT matrix

def plot_sim(S0,T,n, func =asset):
    if func == asset:
        if n > 10 :
            price_list = asset(S0,T,10)
            plt.plot(price_list)
        else:
            price_list = asset(S0,T,n)
            plt.plot(price_list)
    if func == currency:
        if n > 10 :
            price_list = currency(S0,T,10)
            plt.plot(price_list)
        else:
            price_list = currency(S0,T,n)
            plt.plot(price_list)
    if func == expBM:
        if n > 10 :
            price_list = expBM(S0,T,10)
            plt.plot(price_list)
        else:
            price_list = expBM(S0,T,n)
            plt.plot(price_list)
    return plt.show()

    