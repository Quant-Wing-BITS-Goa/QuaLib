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

    def HRP(self):
        estimate_correl = self.df.corr(method='pearson')
        estimate_covar = self.df.cov()
        distances = np.sqrt((1 - estimate_correl) / 2)
        from scipy.cluster.hierarchy import linkage
        link = linkage(estimate_correl, 'single')
        def get_quasi_diag(link):
            link = link.astype(int)
            sort_ix = pd.Series([link[-1,0], link[-1,1]]) 
            num_items = link[-1, 3]
            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
                df0 = sort_ix[sort_ix >= num_items] 
                i = df0.index
                j = df0.values - num_items # 
                sort_ix[i] = link[j,0] 
                df0  = pd.Series(link[j, 1], index=i+1)
                sort_ix = sort_ix.append(df0)
                sort_ix = sort_ix.sort_index()
                sort_ix.index = range(sort_ix.shape[0])
            return sort_ix.tolist()
        sort_ix = get_quasi_diag(link)
        def get_cluster_var(cov, c_items):
            cov_ = cov.iloc[c_items, c_items] 
            ivp = 1./np.diag(cov_)
            ivp/=ivp.sum()
            w_ = ivp.reshape(-1,1)
            c_var = np.dot(np.dot(w_.T, cov_), w_)[0,0]
            return c_var
        def get_rec_bipart(cov, sort_ix):
            w = pd.Series(1, index=sort_ix)
            c_items = [sort_ix]
            while len(c_items) > 0:
                c_items = [i[int(j):int(k)] for i in c_items for j,k in 
                   ((0,len(i)/2),(len(i)/2,len(i))) if len(i)>1]
                for i in range(0, len(c_items), 2):
                    c_items0 = c_items[i] 
                    c_items1 = c_items[i+1]
                    c_var0 = get_cluster_var(cov, c_items0)
                    c_var1 = get_cluster_var(cov, c_items1)
                    alpha = 1 - c_var0/(c_var0+c_var1)
                    w[c_items0] *= alpha
                    w[c_items1] *=1-alpha
            return w
        
        weights = get_rec_bipart(estimate_covar, sort_ix)
        new_index = [self.df.columns[i] for i in weights.index]
        weights.index = new_index
        return weights
        #code the Hierarchical Risk Parity Algorithm here 

    def markowitz(self,port):
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
            def portfolio_stats(weights):
                # Convert to array in case list was passed instead.
                weights = np.array(weights)
                port_return = np.sum(self.df.mean() * weights) * 252
                port_vol = np.sqrt(np.dot(weights.T, np.dot(self.df.cov() * 252, weights)))
                sharpe = port_return/port_vol
                return {'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}
            def minimize_volatility(weights):  
                return portfolio_stats(weights)['volatility'] 
        constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
        bounds = []
        for i in range(len(self.df.columns)):
            bounds.append((0,1))
        init_guess=[]
        for i in range(len(self.df.columns)):
            init_guess.append(1/len(self.df.columns))
        optimal_volatilty=scipy.optimize.minimize(minimize_volatility,init_guess,method = 'SLSQP',bounds = bounds,constraints = constraints)
        result=dict(zip(self.df.columns,optimal_volatilty .x))
        return result
        

    def original():
        #Leave this for now

