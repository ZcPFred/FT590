import numpy as np
import pandas as pd




def chol_psd(sigma):
    root = np.full(sigma.shape, 0.0, dtype="float64")
    n = root.shape[1]
    # loop over columns
    for j in range(n):
        s = 0.0
        # if we are not on the first column, calculate the dot product of the preceeding row values.
        if j > 0:
            s = root[j, 0:(j)] @ root[j, 0:(j)]
        temp = sigma[j][j] - s
        if -1e-8 <= temp <= 0:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        # Check for the 0 eigan value.  Just set the column to 0 if we have one
        if root[j, j] == 0.0:
            continue
        for i in range(j + 1, n):
            root[i, j] = (sigma[i, j] - root[i, :j] @ root[j, :j].T) / root[j, j]

    return np.matrix(root)


# Near PSD Matrix
def near_psd(A, epsilon=0.0):
    n = A.shape[0]
    invSD = None
    out = A.copy()
    # calculate the correlation matrix if we got a covariance
    if sum(np.diag(out) == 1) != n:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = (invSD @ out) @ invSD
    # SVD, update the eigen value and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1 / ((vecs * vecs) @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = (T @ vecs) @ l
    out = B @ B.T

    if invSD != None:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD * out * invSD
    return (out)


def frobenius_norm(A):
    n = len(A)
    s = 0
    for i in range(n):
        for j in range(n):
            s += A[i, j] ** 2
    return s


def proj_u(matrix):
    corr = matrix.copy()
    np.fill_diagonal(corr, 1)
    return corr


def proj_s(matrix):
    eig_vals, eig_vecs = np.linalg.eigh(matrix)
    eig_vals[eig_vals < 0] = 0
    return eig_vecs @ np.diag(eig_vals) @ eig_vecs.T


def higham_nearestPSD(corr, tolerance=1e-9):
    # Δ𝑆0 = 0, 𝑌0 = 𝐴, γ0 = 𝑚𝑎𝑥 𝐹𝑙𝑜𝑎𝑡
    dS = 0
    y = corr
    pre_gamma = float("inf")

    # 𝐿𝑜𝑜𝑝 𝑘 ∈ 1... 𝑚𝑎𝑥 𝐼𝑡𝑒𝑟𝑎𝑡𝑖𝑜𝑛𝑠
    iter = 100
    for i in range(iter):
        r = y - dS  # 𝑅𝑘 = 𝑌𝑘−1 − Δ𝑆𝑘−1
        x = proj_s(r)  # 𝑋𝑘 = 𝑃𝑆(𝑅𝑘)
        dS = x - r  # Δ𝑆𝑘 = 𝑋𝑘 − 𝑅𝑘
        y = proj_u(x)  # 𝑌𝑘 = 𝑃𝑈(𝑋𝑘)
        gamma = frobenius_norm(y - corr)

        # 𝑖𝑓 |γ𝑘−1 − γ𝑘 |< 𝑡𝑜𝑙 𝑡ℎ𝑒𝑛 𝑏𝑟𝑒𝑎𝑘
        if abs(gamma - pre_gamma) < tolerance:
            break
        pre_gamma = gamma

    return y






