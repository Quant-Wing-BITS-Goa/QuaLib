import numpy as np
import pandas as pd

def GARCH(mean, alpha, lam, current, k):
    return (mean**2 + (current**2 - mean**2) * pow(1 - 1/(1 - ((1 - alpha) * (1 - lam))), k))

def c2c():
    #Function goes here. Refer chapter 10 of Paul Wilmott

def parkinson():
    #Function goes here

def garman():
    #Function goes here

def rogers():
    #Function goes here
