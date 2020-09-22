import numpy as np
import pandas as pd
import mibian
from scipy.stats import norm

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
    
    def __init__(self,S,K,r,stdev,T):
        self.S = S
        self.K=K
        self.r=r
        self.stdev = stdev
        self.T=T
    
    def impliedvolatility(self):
        return (mibian.BS([self.S, self.K, self.r, self.T] , callPrice = BSE(self.S, self.K, self.r, self.stdev, self.T).BSM()).impliedVolatility)

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + self.stdev ** 2 / 2) * self.T) / (self.stdev * np.sqrt(self.T))
 
    def d2(self):
        return (np.log(self.S / self.K) + (self.r - self.stdev ** 2 / 2) * self.T) / (self.stdev * np.sqrt(self.T))

    def BSM(self):
        return (self.S * norm.cdf(self.d1())) - (self.K * np.exp(-1*self.r * self.T) * norm.cdf(self.d2()))


#a = BSE(120,100,0.01,0.5,1)
#print("Call option price: ", a.BSM())
