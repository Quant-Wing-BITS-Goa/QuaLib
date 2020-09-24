import numpy as np
import pandas as pd
from scipy.stats import norm
from math import sqrt, pi
import scipy.stats as si

class BSE:
    """
    BSE Class

    Args:
    S - Spot Price
    K - Strike Price
    r - Risk Free Rate
    stdev - Standard deviation of underlying asset
    T - Time of expiry of the option

    """

    def __init__(self,S,K,r,T,stdev = None):
        self.S = S
        self.K=K
        self.r=r
        self.stdev = stdev
        self.T=T

    def impliedvolatility(self, option = 'Call', price = None):
        if price is None:
            if option == 'Put':
                price = BSE(self.S, self.K, self.r, self.stdev, self.T).putprice()
            if option == 'Call':
                price = BSE(self.S, self.K, self.r, self.stdev, self.T).BSM()

        tolerance = 1e-3
        epsilon = 1
        count = 0
        max_iterations = 1000
        volatility = 0.50

        while epsilon > tolerance:
            count += 1

            if count >= max_iterations:
                break
            orig_volatility = volatility

            if option == 'Put':
                function_value  = BSE(self.S, self.K, self.r, volatility, self.T).putprice() - price
            if option == 'Call':
                function_value  = BSE(self.S, self.K, self.r, volatility, self.T).BSM() - price

            volatility += -function_value / (self.S * norm.pdf(self.d1()) * sqrt(self.T))
            epsilon = abs((volatility - orig_volatility) / orig_volatility)

        return (volatility)

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + self.stdev ** 2 / 2) * self.T) / (self.stdev * np.sqrt(self.T))

    def d2(self):
        return (np.log(self.S / self.K) + (self.r - self.stdev ** 2 / 2) * self.T) / (self.stdev * np.sqrt(self.T))

    def delta(self, option = 'Call'):
        if option == 'Put':
            return (-norm.cdf(-self.d1()))
        if option == 'Call':
            return ( norm.cdf( self.d1()))

    def theta(self, option = 'Call'):
        if option == 'Put':
            return (-self.S * self.stdev * norm.pdf(-self.d1()) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()))
        if option == 'Call':
            return (-self.S * self.stdev * norm.pdf( self.d1()) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf( self.d2()))

    def vega(self):
        return (self.S * norm.pdf(self.d1()) * np.sqrt(self.T))

    def gamma(self):
        return (self.K * np.exp(-self.r * self.T) * (norm.pdf(self.d2()) / (self.S**2 * self.stdev * np.sqrt(self.T))))

    def rho(self, option = 'Call'):
        if option == 'Call':
            return ( self.K * self.T * np.exp(-self.r * self.T) * norm.cdf( self.d2()))
        if option == 'Put':
            return (-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2()))

    def BSM(self):
        return (self.S * norm.cdf(self.d1())) - (self.K * np.exp(-1*self.r * self.T) * norm.cdf(self.d2()))

    def putprice(self):
        return (-self.S * norm.cdf(-self.d1())) + (self.K * np.exp(-1*self.r * self.T) * norm.cdf(-self.d2()))

    def premium(self, option="P"):
        if option == "C":
            return self.BSM()
        else:
            return (self.BSM() + self.K/(1+self.r)**self.T - self.S)


a = BSE(120,100,0.01,1,stdev=0.5)
#print("Call option price: ", a.premium("C"))
#print("Put option price: ", a.premium("P"))
#print("Implied Volatility: ", a.impliedvolatility())