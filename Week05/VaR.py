from scipy.stats import norm
from scipy import stats as st
import numpy as np
import pandas as pd
import scipy.optimize as opt
from risk_lib import cov, simulation


def VaR_normal(ret, alpha):
    μ = ret.mean()
    σ = ret.std()
    VaR = -(norm.ppf(alpha) * σ + μ)
    return VaR[0]


def VaR_EwNnormal(ret, alpha, lamb):
    μ = ret.mean()
    σ = np.sqrt(cov.weighted_cov(lamb, ret)[0])
    VaR = -(norm.ppf(alpha) * σ + μ)
    return VaR[0]


def VaR_T(ret, alpha):
    def t_generalized(param_vec, x):
        func = -np.log(st.t(df=param_vec[0], loc=param_vec[1], scale=param_vec[2]).pdf(x)).sum()
        return func
    df, mean, scale = simulation.get_T_optParam(ret)
    VaR= -(scale * st.t.ppf(alpha, df) + mean)
    return VaR


def VaR_His(ret, alpha):
    VaR=-np.percentile(ret, q=alpha * 100)
    return VaR

