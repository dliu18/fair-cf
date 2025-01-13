import numpy as np

from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

import cvxpy as cp
import time

import pickle
from tqdm import tqdm 

from sklearn.metrics import roc_auc_score

from loaders import lastfm, movielens

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
        R_train[u_idx, pos_item_idxs[:num_train]] = R[u_idx, pos_item_idxs[:num_train]]
        if num_val > 0:
            R_val[u_idx, pos_item_idxs[num_train:num_train + num_val]]\
                =R[u_idx, pos_item_idxs[num_train:num_train + num_val]]
        if num_test > 0:
            R_test[u_idx, pos_item_idxs[-num_test:]]\
                = R[u_idx, pos_item_idxs[-num_test:]] 

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
def get_inverse_cdf(x, bins=100):
    '''
        return an array whose i-th entry is the inverse cdf of x[i] within x. 
        The granularity of the inverse cdf is determined by bins.
    '''
    quantiles = [np.quantile(x, q/bins) for q in range(1, bins + 1)]
    output = []
    for i in range(len(x)):
        for j in range(bins):
            if quantiles[j] >= x[i]:
                output.append((j + 1)/bins)
                break
    assert len(output) == len(x)
    return np.array(output)

def get_item_performance(yhat, R_train, R_test, low_cols, med_cols, high_cols, pops):

    def _aucs(yhat, R_train, R_test):
        aucs = []
        mask = []
        all_pop = np.sum(R_train, axis=0)
        for j in range(R_test.shape[1]):
            true_labels = R_test[:, j][R_train[:, j] == 0]
            pred_labels = yhat[:, j][R_train[:, j] == 0]
            if np.sum(true_labels) == 0:
                aucs.append(0)
                mask.append(False)
            else:
                aucs.append(roc_auc_score(true_labels > 0, pred_labels))
                mask.append(True)
        return np.array(aucs), np.array(mask)

    def _mrrs(yhat, R_train, R_test, k=50):
        pred_labels = get_prediction_matrix(yhat.T, R_train.T, R_test.T > 0)
        pred_data = pred_labels[:, :k]
        scores = 1./np.arange(1, k+1)
        pred_data = pred_data * scores
        pred_data = pred_data.sum(1)
        return pred_data, np.array([True] * len(pred_data))

    # metrics, mask = _aucs(yhat, R_train, R_test)
    metrics, mask = _mrrs(yhat, R_train, R_test, k=50)

    high_avg, high_std = np.mean(metrics[high_cols][mask[high_cols]]), np.std(metrics[high_cols][mask[high_cols]])
    med_avg, med_std = np.mean(metrics[med_cols][mask[med_cols]]), np.std(metrics[med_cols][mask[med_cols]])
    low_avg, low_std = np.mean(metrics[low_cols][mask[low_cols]]), np.std(metrics[low_cols][mask[low_cols]])

    pops = pops[mask]
    # pops /= np.max(pops)

    metrics_filtered = metrics[mask]
    cov = np.dot(pops - np.mean(pops), metrics_filtered - np.mean(metrics_filtered))
    # cov = np.dot(pops, metrics_filtered)
    # corr = pearsonr(pops/np.max(pops), metrics[mask])[0]

    return (high_avg, high_std), (med_avg, med_std), (low_avg, low_std), cov

def gini_coef(pops, metric):
    assert len(pops) == len(metric)

    M = len(pops)
    pop_item_order = np.argsort(pops)
    items_ordered_by_pop = metric[pop_item_order]
    coefs = np.array([2*i - M - 1 for i in range(1, M+1)])

    return np.dot(coefs, items_ordered_by_pop) / (M * np.sum(items_ordered_by_pop))

def get_pop_opp_bias(yhat, R_train, R_test, k):
    """
        Unit test in main function
    """
    assert np.all((R_test == 0) + (R_test == 1))
    M = R_test.shape[1]

    pred_matrix = get_prediction_matrix_in_place(yhat, R_train, k)
    true_positives = np.sum(pred_matrix * R_test, axis=0)
    recall_denom = np.array([max(np.sum(R_test[:, j]), 1) for j in range(M)])
    recalls =  true_positives / recall_denom

    pops = np.sum(R_train, axis=0)
    
    return gini_coef(pops, recalls)

def get_specialization(model, r, low_cols, med_cols, high_cols, gamma=0):
    specs = model.get_specialization(r)
    if gamma != 0:
        specs = model.get_specialization(r, gamma=gamma)
    pops = np.sum(model.get_R(), axis=0)
    # pops = get_inverse_cdf(np.sum(model.get_R(), axis=0))
    # pops /= np.max(pops)

    return (specs[high_cols].mean(), specs[high_cols].std()),\
        (specs[med_cols].mean(), specs[med_cols].std()),\
        (specs[low_cols].mean(), specs[low_cols].std()),\
        gini_coef(pops, specs)
        # np.dot(pops - np.mean(pops), specs - np.mean(specs))
        # pearsonr(pops / np.max(pops), specs / np.max(specs))[0]

def get_prediction_matrix_in_place(yhat, R_train, k):
    """
    Returns a prediction matrix of shape nxm where R[i][j] = 1 if the prediction is 1 for user i and item j
    """
    outputs = np.zeros(R_train.shape)
    for i in range(len(yhat)):
        sorted_item_idx = np.argsort(-yhat[i])
        outputs[i][sorted_item_idx[:k]] = 1
    return outputs

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

def fair_pca_max_pred(X, d, r, accuracy_bound = 0, gamma=0, cosine = False, max_iters = -1, verbose = False):
    '''
        Setting cosine to True modifies the objective to approximate the cosine similarity between the predicted and 
        actual values.
    '''
    proj_mtx = cp.Variable((d,d), PSD=True)
    
    # cov_mtx = X.T @ X
    # w, _ = eigsh(cov_mtx, which = "LM", k = r)
    # vanilla_obj_value = np.sum(w)
    
    constraints = [
        np.eye(d) - proj_mtx >> 0,
        cp.trace(proj_mtx) == r,
        # cp.trace(cov_mtx @ proj_mtx) >= accuracy_bound * vanilla_obj_value
    ]
    
    weights = [np.linalg.norm(X[:, j])**gamma for j in range(d)]
    weight_mtx = np.diag(weights)

    coefs = weight_mtx.T @ X.T @ X
    prob = cp.Problem(cp.Minimize(-cp.trace(coefs @ proj_mtx)),
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


if __name__ == "__main__":
    R = movielens.movielens(min_ratings = 1, min_users = 5, binary = True).get_X()
    R[R != 0] = 1
    rng = np.random.default_rng(seed=2020)
    R = R[:, rng.choice(a=R.shape[1], size=1200, replace=False)]

    R_train, R_val, R_test = split_data(R, 
        val_ratio = 0.1, 
        test_ratio = 0.2)

    biases = []
    pops = np.sum(R_train + R_val, axis=0)
    item_idx_order = np.argsort(-pops)
    preds = np.zeros(R_test.shape)
    for i in tqdm(range(len(item_idx_order))):
        item_idx = item_idx_order[i]
        preds[:, item_idx] = R_test[:, item_idx]

        pop_opp_bias = get_pop_opp_bias(preds, R_train + R_val, R_test>0, k=R_test.shape[1])
        biases.append(pop_opp_bias)
    fig, ax = plt.subplots()
    ax.plot(biases)
    ax.set_xlabel("Num of Top Items Selected")
    ax.set_ylabel("Popularity Opportunity Bias")
    fig.savefig("figs/unit_tests/pop_opp_bias.pdf", bbox_inches="tight")
