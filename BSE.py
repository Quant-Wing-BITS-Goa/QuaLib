import numpy as np
import pandas as pd
from scipy.stats import norm
from math import sqrt, pi

class BSE:
    """
    BSE Class
    
    Args:
    S - Spot Price
    K - Strike Price
    r - Risk Free Rate
    stdev - Standard deviation of underlying asset (actual volatility)
    T - Time of expiry of the option

    """

    def __init__(self, S, K, r, T, stdev = None):
        self.S = S
        self.K = K
        self.r = r
        self.stdev = stdev
        self.T = T

    def impliedvolatility(self, option = "Call", price = None):
        if price is None:
            if option == "Put":
                price = BSE(self.S, self.K, self.r, self.stdev, self.T).premium(option = "Put")
            else:
                price = BSE(self.S, self.K, self.r, self.stdev, self.T).premium(option = "Call")

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

            if option == "Put":
                function_value  = BSE(self.S, self.K, self.r, volatility, self.T).premium(option = "Put")  - price
            else:
                function_value  = BSE(self.S, self.K, self.r, volatility, self.T).premium(option = "Call") - price

            volatility += -function_value / (self.S * norm.pdf(self.d1()) * sqrt(self.T))
            epsilon = abs((volatility - orig_volatility) / orig_volatility)

        return volatility

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + self.stdev ** 2 / 2) * self.T) / (self.stdev * np.sqrt(self.T))

    def d2(self):
        return (np.log(self.S / self.K) + (self.r - self.stdev ** 2 / 2) * self.T) / (self.stdev * np.sqrt(self.T))

    def delta(self, option = "Call"):
        if option == "Put":
            return (-norm.cdf(-self.d1()))
        else:
            return norm.cdf(self.d1())

    def theta(self, option = "Call"):
        if option == "Put":
            return (-self.S * self.stdev * norm.pdf(-self.d1()) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()))
        else:
            return (-self.S * self.stdev * norm.pdf( self.d1()) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf( self.d2()))

    def vega(self):
        return (self.S * norm.pdf(self.d1()) * np.sqrt(self.T))

    def gamma(self):
        return (self.K * np.exp(-self.r * self.T) * (norm.pdf(self.d2()) / (self.S**2 * self.stdev * np.sqrt(self.T))))

    def rho(self, option = "Call"):
        if option == "Put":
            return (-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2()))
        else:
            return ( self.K * self.T * np.exp(-self.r * self.T) * norm.cdf( self.d2()))

    def callprice(self):
        return (self.S * norm.cdf(self.d1())) - (self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2()))

    def premium(self, option = "Call"):
        if option == "Put"
            return (self.callprice() + self.K/(1 + self.r)**self.T - self.S) #Using Put-Call parity
        else:
            return self.callprice()

    def hedge(va,vi=None,V,use):
        """
        This function is to be only used when implied volatilty 
        and actual volatility differ. It is used to calculate the amoung
        of hedge one will need to do when there is a volatility 
        difference to make a profit. 

        Function prints amd returns the amount of hedge and profit

        Arguments
        va: Actual Volatility 
        vi: Implied Volatility, if not passed, calculate using the implied volatility function above
        V: Price of the option
        use: Can be either "actual" or "implied" to specify which volatility to use to set up the hedge

        """

        #delta is the amount of hedge and profit is the profit one will make using the specific hedge.
        if vi==None:
            vi = #Call the function to calculate implied volatility here 
        if use=="actual":
            delta = #N(d1)
            profit = #Calculate using formula give in Paul Wilmott Chapter 10

        if use=="implied":
            delta = #N(d1)
            profit = #Check the literature here, no specific answer as such.





a = BSE(120,100,0.01,1,stdev=0.5)
#print("Call option price: ", a.premium("C"))
#print("Put option price: ", a.premium("P"))
