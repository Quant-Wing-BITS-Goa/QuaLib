import numpy as np
import pandas as pd
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
        
    def d1(self):
        return (np.log(self.S / self.K) + (self.r + self.stdev ** 2 / 2) * self.T) / (self.stdev * np.sqrt(self.T))
 
    def d2(self):
        return (np.log(self.S / self.K) + (self.r - self.stdev ** 2 / 2) * self.T) / (self.stdev * np.sqrt(self.T))

    def BSM(self):
        return (self.S * norm.cdf(self.d1())) - (self.K * np.exp(-1*self.r * self.T) * norm.cdf(self.d2()))
    
     def premium(self, option="P"):
        if option == "C":
            return self.BSM()
        else:
            return (self.BSM() + self.K/(1+self.r)**self.T - self.S)


#a = BSE(120,100,0.01,0.5,1)
#print("Call option price: ", a.premium("C"))
#print("Put option price: ", a.premium("P"))
