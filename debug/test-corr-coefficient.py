import sys
import numpy as np
import matplotlib.pyplot as plt
import utils
import models
from scipy.stats import pearsonr

from tqdm import tqdm
from loaders import lastfm, movielens

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

	specs = model_obj.get_specialization(r)
	pops = np.sum(model_obj.get_R(), axis=0)
	pops /= np.max(pops)

	corr = pearsonr(pops, specs)
	cov = np.dot(specs - np.mean(specs), pops - np.mean(pops))

	ax_agg[idx].scatter(pops, specs, alpha=0.3, label=model_name)
	ax_agg[idx].set_xlabel("Popularity")
	ax_agg[idx].set_ylabel("Specialization")
	ax_agg[idx].set_title(model_name)
	ax_agg[idx].set_xscale("log")
	# print(f"Model: {model_name}\t Correlation: {corr[0]}\t p-value:{corr[1]}")
	print(f"Model: {model_name}\t Cov: {cov}")


fig_agg.savefig("figs/specialization_popularity_scatter.pdf", bbox_inches="tight")