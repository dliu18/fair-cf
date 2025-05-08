import sys
import numpy as np
import matplotlib.pyplot as plt
import utils
import models
from scipy.stats import pearsonr

from tqdm import tqdm
from loaders import lastfm, movielens

def get_item_preference_auc_roc(yhat, R_train, R_test, low_cols, med_cols, high_cols):

    def _aucs(yhat, R_train, R_test):
        aucs = []
        mask = []
        all_pop = np.sum(R_train, axis=0)
        for j in range(R_test.shape[1]):
            true_labels = R_test[:, j][R_train[:, j] == 0]
            pred_labels = yhat[:, j][R_train[:, j] == 0]
            if np.sum(true_labels) == 0:
                aucs.append(0)
                mask.append(0)
            else:
                aucs.append(roc_auc_score(true_labels > 0, pred_labels))
                mask.append(1)
        return np.array(aucs), np.array(mask)

    # TODO: clean up this logic
    aucs, mask = _aucs(yhat, R_train, R_test)
    # high_auc_avg, high_auc_std = np.mean(aucs[high_cols][mask[high_cols]]), np.std(aucs[high_cols][mask[high_cols]])
    # med_auc_avg, med_auc_std = np.mean(aucs[med_cols][mask[med_cols]]), np.std(aucs[med_cols][mask[med_cols]])
    # low_auc_avg, low_auc_std = np.mean(aucs[low_cols][mask[low_cols]]), np.std(aucs[low_cols][mask[low_cols]])
    return aucs[low_cols][mask[low_cols]]

obj = lastfm.lastfm()
# setting min_ratings = 10 minimizes the number of users with no test interactions
obj.filter(min_users = 10, min_ratings = 10)
R = obj.get_P()
R = np.linalg.inv(np.diag(np.sum(R, axis=1))) @ R

R_train, R_val, R_test = utils.split_data(R, 
	val_ratio = 0.1, 
	test_ratio = 0.2)

high_cols, med_cols, low_cols = utils.get_popularity_splits(R_train)

dataset_name = "lastfm-explicit"
model_list = [
	("PCA", models.PCA(R_train + R_val)),
	("Normalized PCA", models.NormalizedPCA(R_train + R_val)),
	("Item-Weighted PCA", models.ItemWeightedPCA(R_train + R_val, dataset_name)),
	("Weighted MF", models.MF(R_train + R_val, dataset_name)),
]

fig_agg, ax_agg = plt.subplots(figsize = (7 * len(model_list), 7), ncols=len(model_list))
r = 64

for idx, model_info in enumerate(model_list):
	model_name, model_obj = model_info

	low_scores = get_item_preference_auc_roc(yhat, R_train + R_val, R_test, low_cols, med_cols, high_cols)
	# low_scores = get_item_preference_auc_roc(yhat, np.zeros(R_train.shape), R_train + R_val, low_cols, med_cols, high_cols)
	ax_agg[idx].hist(low_scores, histtype="step", bins=20)
	ax_agg[idx].set_xlabel("Item-Preference AUCs")
	# ax_agg[idx].set_ylabel("Specialization")
	ax_agg[idx].set_title(model_name)
	# print(f"Model: {model_name}\t Correlation: {corr[0]}\t p-value:{corr[1]}")
	print(f"Model: {model_name}\t Num Scores: {len(low_scores)}")


fig_agg.savefig("figs/debug_auc_scores.pdf", bbox_inches="tight")