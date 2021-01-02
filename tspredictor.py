import numpy as np
import pandas as pd
from math import sqrt, pi
from scipy.stats import norm
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error


class tspredictor:

    """
    BSE Class
    
    Args:
    df: Dataframe consisting close price(time series data)

    """
    #Initialize a check to see if statsmodels is initialised. Feel free to use it any of the implementations below

    def __init__(self, df):
        self.df = df

    def AR(self, lags):

        X = self.df.values
        train, test = X[1:int(0.90 * len(X))], X[int(0.90 * len(X)):]
        model = AutoReg(train, lags = lags)
        model_fit = model.fit()

        predictions = model_fit.predict(start = len(train), end = len(train) + len(test) - 1, dynamic = False)
        pred_list = []
        expd_list = []

        for i in range(len(predictions)):
            pred_list.append(predictions[i])
            expd_list.append(test[i])
        
        rmse = sqrt(mean_squared_error(test, predictions))
        print('RSME:', rmse)

        results = pd.DataFrame(np.column_stack([pred_list, expd_list]), columns = ['Predictions', 'Test'])        
        return results
    
    def ARMA(self, order):
        X = self.df.values
        train, test = X[1:int(0.90 * len(X))], X[int(0.90 * len(X)):]
        history = [x for x in train]
        model = ARMA(history, order = order)
        model_fit = model.fit(disp = 0)
        pred_list = []
        expd_list = []

        for i in range(len(test)):
            y_hat = model_fit.forecast()[0]
            pred_list.append(y_hat)
            expd_list.append(test[i])

        error = mean_squared_error(test, pred_list)
        print('Error:', error)

        results = pd.DataFrame(np.column_stack([pred_list, expd_list]), columns = ['Predictions', 'Test'])        
        return results


    def ARIMA(self, order):

        X = self.df.values
        train, test = X[1:int(0.90 * len(X))], X[int(0.90 * len(X)):]
        history = [x for x in train]
        model = ARIMA(history, order = order)
        model_fit = model.fit(disp = 0)
        pred_list = []
        expd_list = []

        for i in range(len(test)):
            y_hat = model_fit.forecast()[0]
            pred_list.append(y_hat)
            expd_list.append(test[i])

        error = mean_squared_error(test, pred_list)
        print('Error:', error)

        results = pd.DataFrame(np.column_stack([pred_list, expd_list]), columns = ['Predictions', 'Test'])        
        return results
    

    def SARIMAX(self, order, seasonal_order):
        X = self.df.values
        train, test = X[1:int(0.90 * len(X))], X[int(0.90 * len(X)):]
        history = [x for x in train]
        model = SARIMAX(history, order = order, seasonal_order = seasonal_order)
        model_fit = model.fit(disp = 0)
        pred_list = []
        expd_list = []

        for i in range(len(test)):
            y_hat = model_fit.forecast()[0]
            pred_list.append(y_hat)
            expd_list.append(test[i])

        error = mean_squared_error(test, pred_list)
        print('Error:', error)

        results = pd.DataFrame(np.column_stack([pred_list, expd_list]), columns = ['Predictions', 'Test'])        
        return results

    def VAR(self, lags = 12, steps=5):
        pass

    def SES(self, alpha = 0.2, init_method = "heuristic"):
        X = self.df.values
        train, test = X[1:int(0.90 * len(X))], X[int(0.90 * len(X)):]
        history = [x for x in train]
        model = SimpleExpSmoothing(history, initialization_method = init_method)
        model_fit = model.fit(smoothing_level = alpha)
        pred_list = []
        expd_list = []

        for i in range(len(test)):
            y_hat = model_fit.forecast()[0]
            pred_list.append(y_hat)
            expd_list.append(test[i])

        error = mean_squared_error(test, pred_list)
        print('Error:', error)

        results = pd.DataFrame(np.column_stack([pred_list, expd_list]), columns = ['Predictions', 'Test'])        
        return results

    def HWES(self, alpha = 0.2, init_method = "heuristic"):
        X = self.df.values
        train, test = X[1:int(0.90 * len(X))], X[int(0.90 * len(X)):]
        history = [x for x in train]
        model = ExponentialSmoothing(history, initialization_method = init_method)
        model_fit = model.fit(smoothing_level = alpha)
        pred_list = []
        expd_list = []

        for i in range(len(test)):
            y_hat = model_fit.forecast()[0]
            pred_list.append(y_hat)
            expd_list.append(test[i])

        error = mean_squared_error(test, pred_list)
        print('Error:', error)

        results = pd.DataFrame(np.column_stack([pred_list, expd_list]), columns = ['Predictions', 'Test'])        
        return results

