import sys
import numpy as np
import matplotlib.pyplot as plt
import utils
import models

from tqdm import tqdm
from loaders import lastfm, movielens

colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']

def _aggregate_performance(dataset_name, R_train, R_val, R_test, k=10, step_size=10):
	model_list = [
		("PCA", models.PCA(R_train + R_val)),
		("Normalized PCA", models.NormalizedPCA(R_train + R_val))
	]
	metrics = ["Recall", "Precision", "NDCG", "MRR"]
	max_r = min(min(R_train.shape), 500)
	rs = np.arange(1, max_r + 1, step_size)
	fig_agg, axs = plt.subplots(ncols = len(metrics), figsize = (7 * len(metrics), 7))

	for idx, model_info in enumerate(model_list):
		model_name, model_obj = model_info
		recalls, ndcgs, precisions, mrrs = [], [], [], []
		for r in tqdm(rs):
			yhat = model_obj.predict_ratings(r)
			yhat_sorted = utils.get_prediction_matrix(yhat, R_train + R_val, R_test)

			recall, precis = utils.RecallPrecision(R_test, yhat_sorted, k)
			recalls.append(recall / len(R_test))
			precisions.append(precis / len(R_test))

			mrrs.append(utils.MRR(yhat_sorted, k) / len(R_test))
			ndcgs.append(utils.NDCG(R_test, yhat_sorted, k) / len(R_test))

		axs[0].plot(rs, recalls, label=model_name, color=colors[idx])
		axs[1].plot(rs, precisions, label=model_name, color=colors[idx])
		axs[2].plot(rs, mrrs, label=model_name, color=colors[idx])
		axs[3].plot(rs, ndcgs, label=model_name, color=colors[idx])

	for idx, ax in enumerate(axs):
		ax.set_xscale("log")
		ax.set_xlabel("Rank")
		ax.set_ylabel(metrics[idx])
		ax.set_title(metrics[idx])
		ax.legend()
		ax.grid()
	fig.savefig("figs/aggregate_performance_%s.pdf" % dataset_name, bbox_width = "tight")

	return

def _performance_by_popularity(dataset_name, R_train, R_val, R_test, step_size=10):
	model_list = [
		("PCA", models.PCA(R_train + R_val)),
		("Normalized PCA", models.NormalizedPCA(R_train + R_val))
	]
	max_r = min(min(R_train.shape), 300)
	rs = np.arange(1, max_r + 1, step_size)
	pop_groups = ["High", "Medium", "Low"]
	fig, axs = plt.subplots(ncols = len(pop_groups), figsize = (7 * len(pop_groups), 7), sharey = True) # corresponding to low, med, high
	low_cols, med_cols, high_cols = utils.get_popularity_splits(R_train + R_val)

	for idx, model_info in enumerate(model_list):
		model_name, model_obj = model_info
		high_auc_avgs, med_auc_avgs, low_auc_avgs = [], [], []
		high_auc_std, med_auc_std, low_auc_std = [], [], []
		for r in tqdm(rs):
			#calculate predicted ratings y-hat
			yhat = model_obj.predict_ratings(r)
			high_auc, med_auc, low_auc = utils.get_item_preference_auc_roc(yhat, R_train + R_val, R_test, low_cols, med_cols, high_cols)
			high_auc_avgs.append(high_auc[0])
			high_auc_std.append(high_auc[1])

			med_auc_avgs.append(med_auc[0])
			med_auc_std.append(med_auc[1])

			low_auc_avgs.append(low_auc[0])
			low_auc_std.append(low_auc[1])

		axs[0].plot(rs, high_auc_avgs, label=model_name, color=colors[idx])
		axs[0].fill_between(rs, 
			np.array(high_auc_avgs) - np.array(high_auc_std), 
			np.array(high_auc_avgs) + np.array(high_auc_std),
			alpha=0.2, color=colors[idx])

		axs[1].plot(rs, med_auc_avgs, label=model_name, color=colors[idx])
		axs[1].fill_between(rs, 
			np.array(med_auc_avgs) - np.array(med_auc_std), 
			np.array(med_auc_avgs) + np.array(med_auc_std),
			alpha=0.2, color=colors[idx])

		axs[2].plot(rs, low_auc_avgs, label=model_name, color=colors[idx])
		axs[2].fill_between(rs, 
			np.array(low_auc_avgs) - np.array(low_auc_std),
			np.array(low_auc_avgs) + np.array(low_auc_std),
			alpha=0.2, color=colors[idx])

	for idx, ax in enumerate(axs):
		ax.set_xscale("log")
		ax.set_xlabel("Rank")
		ax.set_ylabel("Item Preference AUC")
		ax.set_title(pop_groups[idx])
		ax.legend()
		ax.grid()
	fig.savefig("figs/performance_by_popularity_%s.pdf" % dataset_name, bbox_width = "tight")

	return 


if __name__ == "__main__":
	# load and split data
	# Note, we require a minimum of 5 users so that every item in the training split has at least 1 user.
	dataset_name = sys.argv[1]
	R = None
	if dataset_name == "lastfm":
		obj = lastfm.lastfm()
		# setting min_ratings = 10 minimizes the number of users with no test interactions
		obj.filter(min_users = 5, min_ratings = 10)
		R = obj.get_P()
		R[R != 0] = 1
	elif dataset_name == "movielens":
		R = movielens.movielens(min_ratings = 1, min_users = 5, binary = True).get_X()
		R[R != 0] = 1
	else:
		raise NotImplementedError("The %s dataset is not available" % dataset_name)

	R_train, R_val, R_test = utils.split_data(R, 
		val_ratio = 0.1, 
		test_ratio = 0.2)
	print("\
		Number of training interactions: %i \n\
		Number of validation interactions: %i \n\
		Number of test interactions: %i\n\
		Number of items with no train interactions: %i\n\
		Number of users with no test interactions: %i"\
		% (np.sum(R_train), np.sum(R_val), np.sum(R_test),
			np.sum(np.sum(R_train + R_val + R_test, axis=0) == 0),
			np.sum(np.sum(R_test, axis=1) == 0)))

	##### Preliminary Statistics
	high_cols, med_cols, low_cols = utils.get_popularity_splits(R_train)
	print("\
		[High] Num Interactions: %i\t Num Items: %i \n\
		[Medium] Num Interactions: %i\t Num Items: %i \n\
		[Low] Num Interactions: %i\t Num Items: %i \n" % (
				np.sum(R_train[:, high_cols]), len(high_cols),
				np.sum(R_train[:, med_cols]), len(med_cols),
				np.sum(R_train[:, low_cols]), len(low_cols)
			)
		)

	# Distribution of item test interactions
	# Conclusion, many items have no test interactions so item-based metrics are difficult for these datasets
	test_item_interactions = np.sum(R_test, axis=0)
	print([np.percentile(test_item_interactions, percent) for percent in [0, 1, 5, 10, 25, 50, 75, 90, 99]])

	# Distribution of user popularity scores
	item_pop = np.sum(R_train + R_val, axis=0)
	R_train_val = R_train + R_val + R_test
	print(np.min(np.sum(R_train_val, axis = 1)))
	user_scores = [np.mean(item_pop[R_train_val[i] > 0]) for i in range(len(R_train))]
	print([np.percentile(user_scores, percent) for percent in np.arange(0, 100, 10)])


	# _aggregate_performance(dataset_name, R_train, R_val, R_test)

	_performance_by_popularity(dataset_name, R_train, R_val, R_test)
