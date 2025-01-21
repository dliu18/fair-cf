import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import models

from tqdm import tqdm
from loaders import lastfm, movielens

colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
try:
	assert len(sys.argv) == 4
except:
	print("Usage: python main.py dataset_name gamma file_suffix")
global_gamma = float(sys.argv[2])
file_suffix = "_" + str(sys.argv[3])

def _specialization(dataset_name, R_train, R_val, figure_values):
	'''
		R_train and R_val do not need to be binary but R_test needs to be binary
	'''
	model_list = [
		("PCA", models.PCA(R_train + R_val)),
		("Normalized PCA", models.NormalizedPCA(R_train + R_val)),
		("Item-Weighted PCA", models.ItemWeightedPCA(R_train + R_val, dataset_name)),
		# ("Weighted MF", models.MF(R_train + R_val, dataset_name)),
	]
	# if "lastfm" not in dataset_name:
	# 	model_list.append(("LightGCN", models.LightGCN(R_train + R_val, dataset_name)))
	pop_groups = ["High", "Medium", "Low"]
	high_cols, med_cols, low_cols = utils.get_popularity_splits(R_train + R_val)
	fig, axs = plt.subplots(ncols = len(pop_groups), figsize = (7 * len(pop_groups), 7))
	fig_agg, ax_agg = plt.subplots(figsize = (7, 7))

	for idx, model_info in enumerate(model_list):
		model_name, model_obj = model_info
		plt_rs = []
		low_spec_avg, med_spec_avg, high_spec_avg = [], [], []
		low_spec_std, med_spec_std, high_spec_std = [], [], []
		agg_specs = []
		for r in tqdm(rs):
			try:
				if model_name == "Item-Weighted PCA":
					high_spec, med_spec, low_spec, agg_spec = utils.get_specialization(model_obj, r, 
						low_cols, med_cols, high_cols,
						gamma=global_gamma,
						file_suffix=file_suffix)
				else:
					high_spec, med_spec, low_spec, agg_spec = utils.get_specialization(model_obj, r, low_cols, med_cols, high_cols)
			except:
				continue

			high_spec_avg.append(high_spec[0])
			high_spec_std.append(high_spec[1])

			med_spec_avg.append(med_spec[0])
			med_spec_std.append(med_spec[1])

			low_spec_avg.append(low_spec[0])
			low_spec_std.append(low_spec[1])

			agg_specs.append(agg_spec)
			plt_rs.append(r)

		axs[0].plot(plt_rs, high_spec_avg, label=model_name, color=colors[idx], linewidth=2)
		axs[1].plot(plt_rs, med_spec_avg, label=model_name, color=colors[idx], linewidth=2)
		axs[2].plot(plt_rs, low_spec_avg, label=model_name, color=colors[idx], linewidth=2)
		ax_agg.plot(plt_rs, agg_specs, label=model_name, color=colors[idx], linewidth=2)

		figure_values.append({
			"Metric": "Specialization",
			"Algorithm": model_name,
			"Popularity": "High",
			"rs": plt_rs,
			"values": high_spec_avg
			})
		figure_values.append({
			"Metric": "Specialization",
			"Algorithm": model_name,
			"Popularity": "Medium",
			"rs": plt_rs,
			"values": med_spec_avg
			})
		figure_values.append({
			"Metric": "Specialization",
			"Algorithm": model_name,
			"Popularity": "Low",
			"rs": plt_rs,
			"values": low_spec_avg
			})

	for idx, ax in enumerate(axs):
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_xlabel("Rank")
		ax.set_ylabel("Specialization")
		ax.set_title("Specialization for %s Popularity" % pop_groups[idx])
		ax.legend()
		ax.grid()

	ax_agg.set_xscale("log")
	# ax_agg.set_yscale("log")
	ax_agg.set_xlabel("Rank")
	ax_agg.set_ylabel("Specialization Bias")
	ax_agg.set_title("Specialization Bias")
	ax_agg.legend()
	ax_agg.grid()

	fig.savefig("figs/facct_submission/specialization_by_pop_%s.pdf" % dataset_name, bbox_width = "tight")
	# fig_agg.savefig("figs/facct_submission/specialization_%s.pdf" % dataset_name, bbox_width = "tight")

def _aggregate_performance(dataset_name, R_train, R_val, R_test, figure_values, k=20, step_size=10):
	'''
		R_train and R_val do not need to be binary but R_test needs to be binary
	'''
	model_list = [
		("PCA", models.PCA(R_train + R_val)),
		("Normalized PCA", models.NormalizedPCA(R_train + R_val)),
		("Item-Weighted PCA", models.ItemWeightedPCA(R_train + R_val, dataset_name)),
		# ("Weighted MF", models.MF(R_train + R_val, dataset_name)),
	]
	# if "lastfm" not in dataset_name:
	# 	model_list.append(("LightGCN", models.LightGCN(R_train + R_val, dataset_name)))
	metrics = ["Recall", "Precision", "NDCG", "MRR"]
	# max_r = min(min(R_train.shape), 500)
	# rs = np.arange(1, max_r + 1, step_size)
	fig_agg, axs = plt.subplots(ncols = len(metrics), figsize = (7 * len(metrics), 7))
	fig_tradeoff, axs_tradeoff = plt.subplots(figsize = (7, 7))

	for idx, model_info in enumerate(model_list):
		model_name, model_obj = model_info
		plt_rs = []
		recalls, ndcgs, precisions, mrrs, unfairness = [], [], [], [], []
		for r in tqdm(rs):
			try:
				if model_name == "Item-Weighted PCA":
					yhat = model_obj.predict_ratings(r, gamma=global_gamma, file_suffix=file_suffix)
				else:
					yhat = model_obj.predict_ratings(r)
			except:
				continue
			yhat_sorted = utils.get_prediction_matrix(yhat, R_train + R_val, R_test)

			recall, precis = utils.RecallPrecision(R_test, yhat_sorted, k)
			recalls.append(recall / len(R_test))
			precisions.append(precis / len(R_test))

			mrrs.append(utils.MRR(yhat_sorted, k) / len(R_test))
			ndcgs.append(utils.NDCG(R_test, yhat_sorted, k) / len(R_test))

			unfairness.append(utils.get_pop_opp_bias(yhat, R_train + R_val, R_test>0, k))
			plt_rs.append(r)

			if r == 32:
				axs_tradeoff.scatter(recalls[-1], unfairness[-1], label=model_name, color=colors[idx])

		axs[0].plot(plt_rs, recalls, label=model_name, color=colors[idx], linewidth=2)
		axs[1].plot(plt_rs, precisions, label=model_name, color=colors[idx], linewidth=2)
		axs[2].plot(plt_rs, mrrs, label=model_name, color=colors[idx], linewidth=2)
		axs[3].plot(plt_rs, ndcgs, label=model_name, color=colors[idx], linewidth=2)
		# axs[4].plot(rs, unfairness, label=model_name, color=colors[idx], linewidth=2)

		figure_values.append({
			"Metric": "Recall",
			"Algorithm": model_name,
			"Popularity": "Overall",
			"rs": plt_rs,
			"values": recalls
			})
		figure_values.append({
			"Metric": "Precision",
			"Algorithm": model_name,
			"Popularity": "Overall",
			"rs": plt_rs,
			"values": precisions
			})
		figure_values.append({
			"Metric": "MRR",
			"Algorithm": model_name,
			"Popularity": "Overall",
			"rs": plt_rs,
			"values": mrrs
			})
		figure_values.append({
			"Metric": "NDCG",
			"Algorithm": model_name,
			"Popularity": "Overall",
			"rs": plt_rs,
			"values": ndcgs
			})

	for idx, ax in enumerate(axs):
		ax.set_xscale("log")
		ax.set_xlabel("Rank")
		ax.set_ylabel(metrics[idx])
		ax.set_title(metrics[idx])
		ax.legend()
		ax.grid()
	axs_tradeoff.set_xlabel("User-level Recall")
	axs_tradeoff.set_ylabel("Popularity-Opportunity Bias")
	axs_tradeoff.legend()
	axs_tradeoff.grid()
	fig_agg.savefig("figs/facct_submission/aggregate_performance_%s.pdf" % dataset_name, bbox_width = "tight")
	# fig_tradeoff.savefig("figs/facct_submission/tradeoff_%s.pdf" % dataset_name, bbox_width="tight")

def _performance_by_popularity(dataset_name, R_train, R_val, R_test, figure_values, metric_name="Precision", step_size=10, out_sample=True):
	model_list = [
		("PCA", models.PCA(R_train + R_val)),
		("Normalized PCA", models.NormalizedPCA(R_train + R_val)),
		("Item-Weighted PCA", models.ItemWeightedPCA(R_train + R_val, dataset_name)),
		# ("Weighted MF", models.MF(R_train + R_val, dataset_name)),
	]
	# if "lastfm" not in dataset_name:
	# 	model_list.append(("LightGCN", models.LightGCN(R_train + R_val, dataset_name)))

	# max_r = min(min(R_train.shape), 300)
	# rs = np.arange(1, max_r + 1, step_size)
	pop_groups = ["Overall", "High", "Medium", "Low"]
	fig, axs = plt.subplots(ncols = len(pop_groups), figsize = (7 * len(pop_groups), 7), sharey=True) # corresponding to overall, low, med, high
	fig_agg, ax_agg = plt.subplots(figsize = (7, 7)) # corresponding to low, med, high
	high_cols, med_cols, low_cols = utils.get_popularity_splits(R_train + R_val)
	pops = utils.get_inverse_cdf(np.sum(R_train + R_val, axis = 0))

	for idx, model_info in enumerate(model_list):
		model_name, model_obj = model_info
		plt_rs = []
		high_avgs, med_avgs, low_avgs, overall_avgs = [], [], [], []
		high_std, med_std, low_std = [], [], []
		corrs = []
		for r in tqdm(rs):
			#calculate predicted ratings y-hat
			try:
				if model_name == "Item-Weighted PCA":
					yhat = model_obj.predict_ratings(r, gamma=global_gamma, use_diagonal=out_sample, file_suffix=file_suffix)
				else:
					yhat = model_obj.predict_ratings(r, use_diagonal=out_sample)
			except:
				continue
			high, med, low, corr = utils.get_item_performance(yhat, 
				R_train + R_val, R_test, 
				low_cols, med_cols, high_cols,
				pops,
				metric_name)
			if not out_sample:
				high, med, low, corr = utils.get_item_performance(yhat, 
					np.zeros(R_train.shape), 
					R_train + R_val, 
					low_cols, med_cols, high_cols,
					pops,
					metric_name)
			high_avgs.append(high[0])
			high_std.append(high[1])

			med_avgs.append(med[0])
			med_std.append(med[1])

			low_avgs.append(low[0])
			low_std.append(low[1])

			overall_total = len(high_cols) * high[0] + len(med_cols) * med[0] + len(low_cols) * low[0]
			overall_avgs.append(overall_total / R_train.shape[1])

			corrs.append(corr)
			plt_rs.append(r)
		axs[0].plot(plt_rs, overall_avgs, label=model_name, color=colors[idx], linewidth=2)
		axs[1].plot(plt_rs, high_avgs, label=model_name, color=colors[idx], linewidth=2)
		# axs[1].fill_between(rs, 
		# 	np.array(high_avgs) - 0.5 * np.array(high_std), 
		# 	np.array(high_avgs) + 0.5 * np.array(high_std),
		# 	alpha=0.2, color=colors[idx])

		axs[2].plot(plt_rs, med_avgs, label=model_name, color=colors[idx], linewidth=2)
		# axs[2].fill_between(rs, 
		# 	np.array(med_avgs) - 0.5 * np.array(med_std), 
		# 	np.array(med_avgs) + 0.5 * np.array(med_std),
		# 	alpha=0.2, color=colors[idx])

		axs[3].plot(plt_rs, low_avgs, label=model_name, color=colors[idx], linewidth=2)
		# axs[3].fill_between(rs, 
		# 	np.array(low_avgs) - 0.5 * np.array(low_std),
		# 	np.array(low_avgs) + 0.5 * np.array(low_std),
		# 	alpha=0.2, color=colors[idx])
		ax_agg.plot(plt_rs, corrs, label=model_name, color=colors[idx], linewidth=2)

		figure_values.append({
			"Metric": f"Item Unfairness - {metric_name}",
			"Algorithm": model_name,
			"Popularity": "High",
			"rs": plt_rs,
			"values": high_avgs
			})
		figure_values.append({
			"Metric": f"Item Unfairness - {metric_name}",
			"Algorithm": model_name,
			"Popularity": "Medium",
			"rs": plt_rs,
			"values": med_avgs
			})
		figure_values.append({
			"Metric": f"Item Unfairness - {metric_name}",
			"Algorithm": model_name,
			"Popularity": "Low",
			"rs": plt_rs,
			"values": low_avgs
			})
		figure_values.append({
			"Metric": f"Item Unfairness - {metric_name}",
			"Algorithm": model_name,
			"Popularity": "Overall",
			"rs": plt_rs,
			"values": overall_avgs
			})

	for idx, ax in enumerate(axs):
		ax.set_xscale("log")
		ax.set_xlabel("Rank")
		ax.set_ylabel("Item-Level Precision@k")
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
		file_prefix += "_in"
	else:
		file_prefix += "_out"
	fig.savefig("figs/facct_submission/%s_%s_%s.pdf" % (file_prefix, dataset_name, metric_name), bbox_width = "tight")
	# fig_agg.savefig("figs/facct_submission/%s_unfairness_%s.pdf" % (file_prefix, dataset_name), bbox_width = "tight")

	return 

def _performance_by_gamma(dataset_name, R_train, R_val, R_test, figure_values, k=10, step_size=10):
	'''
		R_train and R_val do not need to be binary but R_test needs to be binary
	'''

	metrics = ["Testing User Recall", "Training Item Performance", "Specialization"]
	# max_r = min(min(R_train.shape), 500)
	# rs = np.arange(1, max_r + 1, step_size)
	fig_agg, axs = plt.subplots(ncols = len(metrics), figsize = (7 * len(metrics), 7))

	gammas = np.arange(-2, 0, 0.1)
	gammas = np.concatenate((gammas, np.array([0])))
	r = 32
	model_obj = models.ItemWeightedPCA(R_train + R_val, dataset_name)
	
	testing_user_recall = []
	training_item_metric = []
	specs = []

	pops = utils.get_inverse_cdf(np.sum(R_train + R_val, axis = 0))
	high_cols, med_cols, low_cols = utils.get_popularity_splits(R_train + R_val)

	for gamma in tqdm(gammas):
		yhat = model_obj.predict_ratings(r, gamma=gamma, file_suffix=file_suffix)
		high, med, low, corr = utils.get_item_performance(yhat, 
			np.zeros(R_train.shape), 
			R_train + R_val, 
			low_cols, med_cols, high_cols,
			pops)
		overall_total = len(high_cols) * high[0] + len(med_cols) * med[0] + len(low_cols) * low[0]
		training_item_metric.append(overall_total / R_train.shape[1])

		yhat_sorted = utils.get_prediction_matrix(yhat, R_train + R_val, R_test)
		recall, precis = utils.RecallPrecision(R_test, yhat_sorted, k)
		testing_user_recall.append(recall / len(R_test))

		spec_per_item = model_obj.get_specialization(r, gamma=gamma, file_suffix=file_suffix)
		specs.append(np.mean(spec_per_item))

		# pop_opp_bias = utils.get_pop_opp_bias(yhat, R_train + R_val, R_test>0, k=20)
		# unfairness.append(pop_opp_bias)

	axs[0].plot(gammas, testing_user_recall, linewidth=2, color=colors[0], label="Item Weighted")
	axs[1].plot(gammas, training_item_metric, linewidth=2, color=colors[0], label="Item Weighted")
	axs[2].plot(gammas, specs, linewidth=2, color=colors[0], label="Item Weighted")

	model_list = [
		("PCA", models.PCA(R_train + R_val)),
		("Normalized PCA", models.NormalizedPCA(R_train + R_val)),
	]
	for idx, model_info in enumerate(model_list):
		model_name, model_obj = model_info
		yhat = model_obj.predict_ratings(r)
		yhat_sorted = utils.get_prediction_matrix(yhat, R_train + R_val, R_test)

		recall, precis = utils.RecallPrecision(R_test, yhat_sorted, k)
		recall = recall / len(R_test)
		axs[0].hlines(recall, gammas[0], gammas[-1], linestyles="dashed", label=model_name, color=colors[idx + 1])
		
		high, med, low, corr = utils.get_item_performance(yhat, 
			np.zeros(R_train.shape), 
			R_train + R_val, 
			low_cols, med_cols, high_cols,
			pops)
		overall_total = len(high_cols) * high[0] + len(med_cols) * med[0] + len(low_cols) * low[0]
		overall_avg = overall_total / R_train.shape[1]
		axs[1].hlines(overall_avg, gammas[0], gammas[-1], linestyles="dashed", label=model_name, color=colors[idx + 1])

		spec_per_item = model_obj.get_specialization(r)
		axs[2].hlines(np.mean(spec_per_item), gammas[0], gammas[-1], linestyles="dashed", label=model_name, color=colors[idx + 1])

	for idx, ax in enumerate(axs):
		ax.set_xlabel("gamma")
		ax.set_ylabel(metrics[idx])
		ax.set_title(metrics[idx])
		ax.grid()
		ax.legend()
	fig_agg.savefig("figs/facct_submission/performance_by_gamma_%s.pdf" % dataset_name, bbox_width = "tight")

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

	rs = [2**i for i in range(2, 11)]
	# rs = utils.get_ds(R)
	R_train, R_val, R_test = utils.split_data(R, 
		val_ratio = 0.1, 
		test_ratio = 0.2)
	print("\
		Number of training interactions: %i \n\
		Number of validation interactions: %i \n\
		Number of test interactions: %i\n\
		Number of items with <10 train interactions: %i\n\
		Number of users with no test interactions: %i"\
		% (np.sum(R_train>0), np.sum(R_val>0), np.sum(R_test>0),
			np.sum(np.sum((R_train + R_val) > 0, axis=0) < 10),
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
	# test_item_interactions = np.sum(R_test>0, axis=0)
	# print([np.percentile(test_item_interactions, percent) for percent in [0, 1, 5, 10, 25, 50, 75, 90, 99]])

	# Distribution of user popularity scores
	# item_pop = np.sum(R_train + R_val, axis=0)
	# R_train_val = R_train + R_val + R_test
	# print(np.min(np.sum(R_train_val, axis = 1)))
	# user_scores = [np.mean(item_pop[R_train_val[i] > 0]) for i in range(len(R_train))]
	# print([np.percentile(user_scores, percent) for percent in np.arange(0, 100, 10)])

	figure_values = []
	# _specialization(dataset_name, R_train, R_val, figure_values)
	# _performance_by_popularity(dataset_name, R_train, R_val, R_test > 0, figure_values, metric_name="Precision", out_sample=False)
	# _performance_by_popularity(dataset_name, R_train, R_val, R_test > 0, figure_values, metric_name="AUC", out_sample=False)
	_aggregate_performance(dataset_name, R_train, R_val, R_test > 0, figure_values)

	# Appendix
	# _performance_by_gamma(dataset_name, R_train, R_val, R_test > 0, figure_values)

	df = pd.DataFrame(figure_values)
	df.to_csv("figs/facct_submission/figure_values_downstream.csv")
