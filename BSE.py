#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from scipy.stats import norm


# In[3]:


class BSE:
    """
    BSE Class
    
    Args:
    S - Spot Price
    K - Strike Price 
    r- Risk Free Rate
    T - Time of expiry of the option
    
    """
    
    def __init__(self,S,K,r,T):
        self.S = S
        self.K=K
        self.r=r
        self.T=T
        
    def d1(self):
        #return (np.log(S / K) + (r + stdev ** 2 / 2) * T) / (stdev * np.sqrt(T))
 
    def d2(self):
        #return (np.log(S / K) + (r - stdev ** 2 / 2) * T) / (stdev * np.sqrt(T))

    def BSM(self):
        #return (S * norm.cdf(d1())) - (K * np.exp(-r * T) * norm.cdf(d2()))


# In[ ]:




