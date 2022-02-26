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


def weighted_pair(x, y, weight):
    n = len(weight)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov = 0
    for i in range(n):
        cov += weight[n - 1 - i] * (x[i] - mean_x) * (y[i] - mean_y)
    return cov


def calculate_weight(lamb, df):
    X = df.index.values
    weight = [(1 - lamb) * lamb ** (i - 1) for i in X]
    weight_adjust = [weight[i] / sum(weight) for i in X]
    return weight_adjust


def weighted_cov(lamb, df):
    n = df.shape[1]
    T = len(df)
    weight = calculate_weight(lamb, df)
    cov_mat = pd.DataFrame(np.zeros((n, n)))
    for i in range(n):
        x = df.iloc[:, i]
        cov_mat.iloc[i, i] = weighted_pair(x, x, weight)
        for j in range(i + 1):
            y = df.iloc[:, j]
            cov_mat.iloc[i, j] = weighted_pair(x, y, weight)
            cov_mat.iloc[j, i] = cov_mat.iloc[i, j]

    return np.array(cov_mat)

