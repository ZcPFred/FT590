import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.optimize as opt
from scipy import stats as st
from risk_lib import psd



def get_Norm_simulation(df, sample_size):
    μ = df.mean()
    σ = df.std()
    result = norm.rvs(loc= μ, scale=σ, size=sample_size)
    return result

def get_T_optParam(df):
    def t_generalized(param_vec, x):
        func = -np.log(st.t(df=param_vec[0], loc=param_vec[1], scale=param_vec[2]).pdf(x)).sum()
        return func
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2},
                {'type': 'ineq', 'fun': lambda x: x[2]})

    data, mean, scale = opt.minimize(fun=t_generalized,
                                       x0=[2, np.array(df).mean(), np.array(df).std()],
                                       constraints=cons,
                                       args=(np.array(df))).x

    return data, mean, scale

def copula(ret, nOfDarws):
    n=ret.shape[1]
    stock_cdf = pd.DataFrame()
    t_params = []

    for col in ret.columns:
        ret[col] -= ret[col].mean()
        df, mean, scale = get_T_optParam(ret[col])
        t_params.append([df, mean, scale])
        stock_cdf[col] = st.t.cdf(ret[col], df=df, loc=mean, scale=scale)

    numOfDraw=1000
    Corr_spearman = st.stats.spearmanr(stock_cdf)[0]

    cholesky = psd.chol_psd(Corr_spearman)
    simuNormal = pd.DataFrame(st.norm.rvs(size=(n,nOfDarws)))
    simulatedT= (cholesky @ simuNormal).T

    Simu_data = pd.DataFrame()
    for i in range(n):
        simu = st.norm.cdf(simulatedT.iloc[:, i])
        Simu_data[ret.columns[i]] = st.t.ppf(simu, df=t_params[i][0], loc=t_params[i][1], scale=t_params[i][2])

    return Simu_data
