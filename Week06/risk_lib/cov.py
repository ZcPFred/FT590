import numpy as np
import pandas as pd

def populateWeights(df, λ):
    n = df.shape[0]
    x = [0.0] * n
    w = [0.0] * n
    cw = [0.0] * n
    n = len(x)
    tw = 0.0
    for i in range(1, n + 1):
        x[i - 1] = i
        w[i - 1] = (1 - λ) * λ ** i
        tw += w[i - 1]
        cw[i - 1] = tw

    for i in range(1, n + 1):
        w[i - 1] = w[i - 1] / tw
        cw[i - 1] = cw[i - 1] / tw
    return w


def out_put_EWCM(df, λ):
    n = df.shape[1]
    w = populateWeights(df, λ)
    cov_matrix = np.zeros([n, n])
    for i in range(df.shape[0]):
        for j in range(n):
            df.iloc[i, j] = df.iloc[i, j] - np.mean(df.iloc[:, j])

    for i in range(n):
        for j in range(n):
            temp = w * df.iloc[:, i]
            cov_matrix[i, j] = np.dot(temp, df.iloc[:, j])

    return cov_matrix

