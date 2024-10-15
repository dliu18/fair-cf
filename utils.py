import numpy as np

from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

import cvxpy as cp
import time

import pickle
from tqdm import tqdm 

def reconstruction_error(X, X_approx, idxs):
    """
    Calculate the average reconstruction error for a set of indices 
    Returns |X[idxs] - X_approx[idxs]|_F^2
    """
    assert len(idxs) != 0
    return np.linalg.norm(X[idxs] - X_approx[idxs]) / len(idxs)

def top_item_accuracy(X, U):
    """
    X is a ratings matrix of users (rows) and items and U is a projection matrix. Returns the fraction of rows for which X and X_approx agree on which item is rated the highest. 
    """
    X_approx = X @ U @ U.T
    return np.sum(np.argmax(X, axis = 1) == np.argmax(X_approx, axis = 1)) / len(X)

#### PCA Lambdas
def vanilla_pca(X, r):
    cov = X.T @ X
    w, v = eigsh(cov, k = r, which = "LM")
    proj = v @ v.T
    return proj

def fair_pca_max_pred(X, d, r, accuracy_bound = 0, cosine = False, max_iters = -1, verbose = False):
    '''
        Setting cosine to True modifies the objective to approximate the cosine similarity between the predicted and 
        actual values.
    '''
    proj_mtx = cp.Variable((d,d), PSD=True)
    
    cov_mtx = X.T @ X
    w, _ = eigsh(cov_mtx, which = "LM", k = r)
    vanilla_obj_value = np.sum(w)
    
    constraints = [
        np.eye(d) - proj_mtx >> 0,
        cp.trace(proj_mtx) == r,
        cp.trace(cov_mtx @ proj_mtx) >= accuracy_bound * vanilla_obj_value
    ]
    
    n = len(X)
    weight_mtx = np.zeros((n, d))
    for j in range(d):
        norm = np.linalg.norm(X[:, j])
        if cosine:
            norm = np.linalg.norm(X[:, j] != 0)
        for i in range(n):
            if X[i, j] > 0:
                weight_mtx[i, j] = 1 / norm
            elif X[i, j] < 0:
                weight_mtx[i, j] = -1 / norm
#             if cosine:
#                 weight_mtx[i, j] *= abs(X[i, j])
    prob = cp.Problem(cp.Minimize(-cp.trace(weight_mtx.T @ (X @ proj_mtx))),
                      constraints)
    if max_iters > 0:
        prob.solve(solver = cp.SCS, max_iters = max_iters, verbose = verbose)
    else:
        prob.solve(solver = cp.SCS)
        
    if prob.status in ["infeasible", "unbounded"]:
        print("SDP Problem is {}".format(prob.status))
        return np.zeros((d, d))
    
    proj = proj_mtx.value
    return proj
##### EVALUATION HELPERS
'''
In general, the below lambdas take as input X, Xs (sub matrices of X) and a projection matrix P
'''

def total_reconstruction_error(X, P):
    return np.linalg.norm(X - X @ P)

def precision_at_k(X, P, k = 10):
    X_approx = X @ P
    d = X.shape[0]
    total_correct = 0
    for i in range(len(X)):
        predicted = np.zeros(d)
        actual = np.zeros(d)
        
        actual[np.argsort(X[i])[-k:]] = 1
        predicted[np.argsort(X_approx[i])[-k:]] = 1
        total_correct += np.dot(actual, predicted)
    return total_correct / len(X)

def max_group_reconstruction(Xs, P, r):
    max_error = -1
    for X in Xs:
        U, sigma, Vh = np.linalg.svd(X, full_matrices = False)
        X_hat = U[:, :r] @ np.diag(sigma[:r]) @ Vh[:r]
        n_g = len(X)
        
        error = (np.linalg.norm(X_hat)**2 - np.trace(X.T @ X @ P)) / n_g
        if error > max_error:
            max_error = error
    return max_error

def weighted_reconstruction_error(Xs, P):
    total_error = 0
    for X in Xs:
        total_error += (np.linalg.norm(X - X @ P) / len(X))
    return total_error / len(Xs)
    
def nash_obj(Xs, P):
    objs = [np.log(np.trace(X.T @ X @ P)) for X in Xs]
    return np.sum(objs)

def update_metrics(X, Xs, proj, r, results, algorithm_name):
    results[algorithm_name]["Reconstruction Error"]["metric"].append(total_reconstruction_error(X, proj))
    results[algorithm_name]["Reconstruction Error"]["r"].append(r)

    results[algorithm_name]["Top-k Precision"]["metric"].append(precision_at_k(X, proj))
    results[algorithm_name]["Top-k Precision"]["r"].append(r)
    
    results[algorithm_name]["Max Group Error"]["metric"].append(max_group_reconstruction(Xs, proj, r))
    results[algorithm_name]["Max Group Error"]["r"].append(r)

    results[algorithm_name]["Weighted Reconstruction Error"]["metric"].append(weighted_reconstruction_error(Xs, proj))
    results[algorithm_name]["Weighted Reconstruction Error"]["r"].append(r)

    results[algorithm_name]["Nash Error"]["metric"].append(nash_obj(Xs, proj))
    results[algorithm_name]["Nash Error"]["r"].append(r)
