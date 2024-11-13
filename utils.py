import numpy as np

from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

import cvxpy as cp
import time

import pickle
from tqdm import tqdm 

from sklearn.metrics import roc_auc_score

#### data preprocessing
def split_data(R, val_ratio = 0.1, test_ratio = 0.2, seed = 1024):
    '''
        Given a ratings matrix R (m users x n items), divides the positive cases into
        training/val/test splits. Outputs the splits as three new ratins matrices:
        R_train, R_val, R_test. 

        The split is done at the user level such that for each user, the interacted items are split.
    '''
    R_train = np.zeros(R.shape)
    R_val = np.zeros(R.shape)
    R_test = np.zeros(R.shape)

    rng = np.random.default_rng(seed=seed)
    for u_idx in range(len(R)):
        pos_item_idxs = np.nonzero(R[u_idx])[0]

        num_positives = len(pos_item_idxs)
        num_val = int(val_ratio * num_positives)
        num_test = int(test_ratio * num_positives)
        num_train = num_positives - num_val - num_test
        assert num_train + num_val + num_test == num_positives

        rng.shuffle(pos_item_idxs)
        R_train[u_idx, pos_item_idxs[:num_train]] = 1.0
        if num_val > 0:
            R_val[u_idx, pos_item_idxs[num_train:num_train + num_val]] = 1.0
        if num_test > 0:
            R_test[u_idx, pos_item_idxs[-num_test:]] = 1.0

    R_composite = R_train + R_val + R_test
    assert np.all(np.sum(R_composite, axis = 0) == np.sum(R, axis = 0))
    assert np.all(np.sum(R_composite, axis = 1) == np.sum(R, axis = 1))

    return R_train, R_val, R_test

def get_popularity_splits(R):
    """
    Returns the column indices that correspond to the high, medium, and low popularity categories. 
    The categories are defined such that each has a third of all interactions. 
    """
    sorted_item_idxs = np.argsort(-np.sum(R, axis = 0))
    R_sorted = R[:, sorted_item_idxs]
    total_interactions = np.sum(R)
    medium_start = -1
    low_start = -1

    j = 0
    interactions_processed = 0
    while low_start < 0:
        interactions_processed += np.sum(R_sorted[:, j])
        if medium_start < 0 and interactions_processed >= total_interactions / 10.0: 
            medium_start = j + 1
        elif low_start < 0 and interactions_processed >= 9 * total_interactions /10.0:
            low_start = j + 1
        j += 1
    
    return sorted_item_idxs[:medium_start], sorted_item_idxs[medium_start:low_start], sorted_item_idxs[low_start:]

#### evaluation metrics
def get_item_preference_auc_roc(yhat, R_train, R_test, low_cols, med_cols, high_cols):

    def _auc_for_group(yhat, R_train, R_test, cols):
        aucs = []
        for j in cols:
            true_labels = R_test[:, j][R_train[:, j] == 0]
            pred_labels = yhat[:, j][R_train[:, j] == 0]
            if np.sum(true_labels) == 0:
                continue
            aucs.append(roc_auc_score(true_labels, pred_labels))
        return np.mean(aucs), np.std(aucs)

    high_auc_avg, high_auc_std = _auc_for_group(yhat, R_train, R_test, high_cols)
    med_auc_avg, med_auc_std = _auc_for_group(yhat, R_train, R_test, med_cols)
    low_auc_avg, low_auc_std = _auc_for_group(yhat, R_train, R_test, low_cols)

    return (high_auc_avg, high_auc_std), (med_auc_avg, med_auc_std), (low_auc_avg, low_auc_std)

def get_prediction_matrix(yhat, R_train, R_test):
    """
    Return a prediction ratings matrix where for each user (row), the ground truth labels are sorted based on the prediction scores
    in yhat. The items corresponding to positives in the training set are excluded.
    """
    yhat[R_train > 0] = -np.inf
    sorted_pred = np.zeros(yhat.shape)
    for i in range(len(yhat)):
        sorted_item_idx = np.argsort(-yhat[i])
        sorted_pred[i] = R_test[i, sorted_item_idx]

    assert np.all(np.sum(sorted_pred, axis = 1) == np.sum(R_test, axis = 1))
    return sorted_pred

def RecallPrecision(R_test, pred_labels, k):
    """
    Recall and Precision
    If a user has no test set interactions, the precision and recall are both 0
    """
    right_pred = pred_labels[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([max(np.sum(R_test[i]), 1) for i in range(len(R_test))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return recall, precis


def MRR(pred_labels, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = pred_labels[:, :k]
    scores = 1./np.arange(1, k+1)
    pred_data = pred_data * scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCG(R_test, pred_labels, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(pred_labels) == len(R_test)
    pred_data = pred_labels[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, ratings in enumerate(R_test):
        length = k if k <= np.sum(ratings) else int(np.sum(ratings))
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

####
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
