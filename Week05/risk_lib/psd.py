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
    # Ξπ0 = 0, π0 = π΄, Ξ³0 = πππ₯ πΉππππ‘
    dS = 0
    y = corr
    pre_gamma = float("inf")

    # πΏπππ π β 1... πππ₯ πΌπ‘ππππ‘ππππ 
    iter = 100
    for i in range(iter):
        r = y - dS  # ππ = ππβ1 β Ξππβ1
        x = proj_s(r)  # ππ = ππ(ππ)
        dS = x - r  # Ξππ = ππ β ππ
        y = proj_u(x)  # ππ = ππ(ππ)
        gamma = frobenius_norm(y - corr)

        # ππ |Ξ³πβ1 β Ξ³π |< π‘ππ π‘βππ πππππ
        if abs(gamma - pre_gamma) < tolerance:
            break
        pre_gamma = gamma

    return y






