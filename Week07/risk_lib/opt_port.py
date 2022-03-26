from scipy.stats import norm
from scipy import stats as st
import numpy as np
import pandas as pd
from scipy import optimize as opt
from risk_lib import cov, simulation

def getrisk(ret, cov, targetRets):
    riskVars = []
    def form(initialParams):
        weights = initialParams
        return weights.T @ cov @ weights
    initialParams = np.array([1 / len(cov)] * len(cov))
    
    for targetRet in targetRets:
        cons = ({'type': 'eq', 'fun': lambda x:  sum(x) - 1},
                {'type': 'eq', 'fun': lambda x:  (x * ret).sum() - targetRet})
        weights = opt.minimize(form, x0=initialParams, constraints=cons).x
        riskVar = np.sqrt(form(weights))
        riskVars.append(riskVar)
    return riskVars


def eff(targetRets, riskVars, rf):
    maxSharpe = 0
    maxIndex = -1
    for i in range(len(targetRets)):
        sharpe = (targetRets[i] - rf) / riskVars[i]
        if sharpe > maxSharpe:
            maxSharpe = sharpe
            maxIndex = i
    return maxSharpe, maxIndex

