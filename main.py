import sys
import numpy as np
import matplotlib.pyplot as plt
import utils
import models

from tqdm import tqdm
from loaders import lastfm, movielens

colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
rs = [2**i for i in range(1, 11)]

def _specialization(dataset_name, R_train, R_val):
	'''
		R_train and R_val do not need to be binary but R_test needs to be binary
	'''
	model_list = [
		("PCA", models.PCA(R_train + R_val)),
		("Normalized PCA", models.NormalizedPCA(R_train + R_val)),
		("Item-Weighted PCA", models.ItemWeightedPCA(R_train + R_val, dataset_name)),
		("Weighted MF", models.MF(R_train + R_val, dataset_name)),
	]
	if "lastfm" not in dataset_name:
		model_list.append(("LightGCN", models.LightGCN(R_train + R_val, dataset_name)))
	pop_groups = ["High", "Medium", "Low"]
	fig, axs = plt.subplots(ncols = len(pop_groups), figsize = (7 * len(pop_groups), 7), sharey=True)
	fig_agg, ax_agg = plt.subplots(figsize = (7, 7))

	for idx, model_info in enumerate(model_list):
		model_name, model_obj = model_info
		low_spec_avg, med_spec_avg, high_spec_avg = [], [], []
		low_spec_std, med_spec_std, high_spec_std = [], [], []
		agg_specs = []
		for r in tqdm(rs):
			high_spec, med_spec, low_spec, agg_spec = utils.get_specialization(model_obj, r, low_cols, med_cols, high_cols)

			high_spec_avg.append(high_spec[0])
			high_spec_std.append(high_spec[1])

			med_spec_avg.append(med_spec[0])
			med_spec_std.append(med_spec[1])

			low_spec_avg.append(low_spec[0])
			low_spec_std.append(low_spec[1])

			agg_specs.append(agg_spec)

		axs[0].plot(rs, high_spec_avg, label=model_name, color=colors[idx], linewidth=2)
		axs[1].plot(rs, med_spec_avg, label=model_name, color=colors[idx], linewidth=2)
		axs[2].plot(rs, low_spec_avg, label=model_name, color=colors[idx], linewidth=2)
		ax_agg.plot(rs, agg_specs, label=model_name, color=colors[idx], linewidth=2)

	for idx, ax in enumerate(axs):
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_xlabel("Rank")
		ax.set_ylabel(pop_groups[idx])
		ax.set_title("Specialization by Popularity Group")
		ax.legend()
		ax.grid()

	ax_agg.set_xscale("log")
	ax_agg.set_xlabel("Rank")
	ax_agg.set_ylabel("Specialization Coefficient")
	ax_agg.set_title("Specialization Correlatation Coefficient")
	ax_agg.legend()
	ax_agg.grid()

	fig.savefig("figs/specialization_by_pop_%s.pdf" % dataset_name, bbox_width = "tight")
	fig_agg.savefig("figs/specialization_%s.pdf" % dataset_name, bbox_width = "tight")

def _aggregate_performance(dataset_name, R_train, R_val, R_test, k=10, step_size=10):
	'''
		R_train and R_val do not need to be binary but R_test needs to be binary
	'''
	model_list = [
		("PCA", models.PCA(R_train + R_val)),
		("Normalized PCA", models.NormalizedPCA(R_train + R_val)),
		("Item-Weighted PCA", models.ItemWeightedPCA(R_train + R_val, dataset_name)),
		("Weighted MF", models.MF(R_train + R_val, dataset_name)),
	]
	if "lastfm" not in dataset_name:
		model_list.append(("LightGCN", models.LightGCN(R_train + R_val, dataset_name)))
	metrics = ["Recall", "Precision", "NDCG", "MRR"]
	# max_r = min(min(R_train.shape), 500)
	# rs = np.arange(1, max_r + 1, step_size)
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

		axs[0].plot(rs, recalls, label=model_name, color=colors[idx], linewidth=2)
		axs[1].plot(rs, precisions, label=model_name, color=colors[idx], linewidth=2)
		axs[2].plot(rs, mrrs, label=model_name, color=colors[idx], linewidth=2)
		axs[3].plot(rs, ndcgs, label=model_name, color=colors[idx], linewidth=2)

	for idx, ax in enumerate(axs):
		ax.set_xscale("log")
		ax.set_xlabel("Rank")
		ax.set_ylabel(metrics[idx])
		ax.set_title(metrics[idx])
		ax.legend()
		ax.grid()
	fig_agg.savefig("figs/aggregate_performance_%s.pdf" % dataset_name, bbox_width = "tight")

	return

def _performance_by_popularity(dataset_name, R_train, R_val, R_test, step_size=10, out_sample=True):
	model_list = [
		("PCA", models.PCA(R_train + R_val)),
		("Normalized PCA", models.NormalizedPCA(R_train + R_val)),
		("Item-Weighted PCA", models.ItemWeightedPCA(R_train + R_val, dataset_name)),
		("Weighted MF", models.MF(R_train + R_val, dataset_name)),
	]
	if "lastfm" not in dataset_name:
		model_list.append(("LightGCN", models.LightGCN(R_train + R_val, dataset_name)))

	# max_r = min(min(R_train.shape), 300)
	# rs = np.arange(1, max_r + 1, step_size)
	pop_groups = ["High", "Medium", "Low"]
	fig, axs = plt.subplots(ncols = len(pop_groups), figsize = (7 * len(pop_groups), 7), sharey = True) # corresponding to low, med, high
	fig_agg, ax_agg = plt.subplots(figsize = (7, 7)) # corresponding to low, med, high
	low_cols, med_cols, high_cols = utils.get_popularity_splits(R_train + R_val)

	for idx, model_info in enumerate(model_list):
		model_name, model_obj = model_info
		high_auc_avgs, med_auc_avgs, low_auc_avgs = [], [], []
		high_auc_std, med_auc_std, low_auc_std = [], [], []
		corrs = []
		for r in tqdm(rs):
			#calculate predicted ratings y-hat
			yhat = model_obj.predict_ratings(r)
			high_auc, med_auc, low_auc, corr = utils.get_item_preference_auc_roc(yhat, R_train + R_val, R_test, low_cols, med_cols, high_cols)
			if not out_sample:
				high_auc, med_auc, low_auc, corr = utils.get_item_preference_auc_roc(yhat, np.zeros(R_train.shape), R_train + R_val, low_cols, med_cols, high_cols)
			high_auc_avgs.append(high_auc[0])
			high_auc_std.append(high_auc[1])

			med_auc_avgs.append(med_auc[0])
			med_auc_std.append(med_auc[1])

			low_auc_avgs.append(low_auc[0])
			low_auc_std.append(low_auc[1])

			corrs.append(corr)
		axs[0].plot(rs, high_auc_avgs, label=model_name, color=colors[idx], linewidth=2)
		# axs[0].fill_between(rs, 
		# 	np.array(high_auc_avgs) - 0.5 * np.array(high_auc_std), 
		# 	np.array(high_auc_avgs) + 0.5 * np.array(high_auc_std),
		# 	alpha=0.2, color=colors[idx])

		axs[1].plot(rs, med_auc_avgs, label=model_name, color=colors[idx], linewidth=2)
		# axs[1].fill_between(rs, 
		# 	np.array(med_auc_avgs) - 0.5 * np.array(med_auc_std), 
		# 	np.array(med_auc_avgs) + 0.5 * np.array(med_auc_std),
		# 	alpha=0.2, color=colors[idx])

		axs[2].plot(rs, low_auc_avgs, label=model_name, color=colors[idx], linewidth=2)
		# axs[2].fill_between(rs, 
		# 	np.array(low_auc_avgs) - 0.5 * np.array(low_auc_std),
		# 	np.array(low_auc_avgs) + 0.5 * np.array(low_auc_std),
		# 	alpha=0.2, color=colors[idx])
		ax_agg.plot(rs, corrs, label=model_name, color=colors[idx], linewidth=2)

	for idx, ax in enumerate(axs):
		ax.set_xscale("log")
		ax.set_xlabel("Rank")
		ax.set_ylabel("Item Preference AUC")
		ax.set_title(pop_groups[idx])
		ax.legend()
		ax.grid()
	ax_agg.set_xscale("log")
	ax_agg.set_xlabel("Rank")
	ax_agg.set_ylabel("Item Unfairness")
	ax_agg.set_title("Item Unfairness")
	ax_agg.legend()
	ax_agg.grid()

	file_prefix = "performance"
	if not out_sample:
		file_prefix += "_out"
	else:
		file_prefix += "_in"
	fig.savefig("figs/%s_%s.pdf" % (file_prefix, dataset_name), bbox_width = "tight")
	fig_agg.savefig("figs/%s_agg_%s.pdf" % (file_prefix, dataset_name), bbox_width = "tight")

	return 


if __name__ == "__main__":
	# load and split data
	# Note, we require a minimum of 5 users so that every item in the training split has at least 1 user.
	dataset_name = sys.argv[1]
	R = None
	if dataset_name == "lastfm":
		obj = lastfm.lastfm()
		# setting min_ratings = 10 minimizes the number of users with no test interactions
		obj.filter(min_users = 10, min_ratings = 10)
		R = obj.get_P()
		R[R != 0] = 1
	elif dataset_name == "lastfm-explicit":
		obj = lastfm.lastfm()
		# setting min_ratings = 10 minimizes the number of users with no test interactions
		obj.filter(min_users = 10, min_ratings = 10)
		R = obj.get_P()
		R = np.linalg.inv(np.diag(np.sum(R, axis=1))) @ R
	elif sys.argv[1] == "lastfm-small":
		obj = lastfm.lastfm()
		# setting min_ratings = 10 minimizes the number of users with no test interactions
		obj.kdd_filter(min_users = 50, min_ratings = 20)
		R = obj.get_P()
		R = np.linalg.inv(np.diag(np.sum(R, axis=1))) @ R
	elif dataset_name == "movielens":
		R = movielens.movielens(min_ratings = 1, min_users = 5, binary = True).get_X()
		R[R != 0] = 1
		rng = np.random.default_rng(seed=2020)
		R = R[:, rng.choice(a=R.shape[1], size=1200, replace=False)]
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
		% (np.sum(R_train>0), np.sum(R_val>0), np.sum(R_test>0),
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
	test_item_interactions = np.sum(R_test>0, axis=0)
	print([np.percentile(test_item_interactions, percent) for percent in [0, 1, 5, 10, 25, 50, 75, 90, 99]])

	# Distribution of user popularity scores
	item_pop = np.sum(R_train + R_val, axis=0)
	R_train_val = R_train + R_val + R_test
	print(np.min(np.sum(R_train_val, axis = 1)))
	user_scores = [np.mean(item_pop[R_train_val[i] > 0]) for i in range(len(R_train))]
	print([np.percentile(user_scores, percent) for percent in np.arange(0, 100, 10)])

	# _specialization(dataset_name, R_train, R_val)
	# _performance_by_popularity(dataset_name, R_train, R_val, R_test > 0, out_sample=False)
	_performance_by_popularity(dataset_name, R_train, R_val, R_test > 0)
	# _aggregate_performance(dataset_name, R_train, R_val, R_test > 0)
