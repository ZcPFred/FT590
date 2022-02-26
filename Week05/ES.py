import numpy as np
import pandas as pd

def ES(ret, alpha):
    VaR_t = -np.percentile(ret, alpha*100)
    ret_alphatail= ret[ret < -VaR_t]
    ES = ret_alphatail.mean()
    return -ES
