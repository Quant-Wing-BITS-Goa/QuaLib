import numpy as np
import pandas as pd
import math

def GARCH(mean, alpha, lam, current, k):
	v = 1/(1 - ((1 - alpha) * (1 - lam)))
	return (pow(mean, 2) + (pow(current, 2) - pow(mean, 2)) * pow(1 - v, k))	

def c2c(df):
    n = len(df.axes[0])
    t = pow(math.log(df['Close'][1]/df['Close'][0]), 2) - ((pow(math.log(df['Close'][1]/df['Close'][0]), 2))/n * (n - 1))
    for i in range(0, n-1):
    	if i != 0:
    		t += pow(math.log(df['Close'][i]/df['Close'][i - 1]), 2) - ((pow(math.log(df['Close'][i]/df['Close'][0]), 2))/n * (n - 1))

    return (math.sqrt(252 * t/(n-1)))

def parkinson(df):
	n = len(df.axes[0])
	t = 0
	for i in df.index:
		t += pow(math.log(df['High'][i]/df['Low'][i]), 2)

	return (math.sqrt(252 * t/(4 * n * math.log(2))))

def garmanklass(df):
	n = len(df.axes[0])
	t = 0
	for i in df.index:
		t += 0.5 * pow(math.log(df['High'][i]/df['Low'][i]), 2) - (2 * math.log(2) - 1) * (pow(math.log(df['Close'][i]/df['Open'][i]), 2))

	return (math.sqrt(252 * t/n))

def rogerssatchell(df):
	n = len(df.axes[0])
	t = 0
	for i in df.index:
		t += math.log(df['High'][i]/df['Close'][i]) * math.log(df['High'][i]/df['Open'][i]) + math.log(df['Low'][i]/df['Close'][i]) * math.log(df['Low'][i]/df['Open'][i])

	return (math.sqrt(252 * t/n))