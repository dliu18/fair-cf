import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

import multiprocessing
from functools import partial
import sys 
import os 

import cvxpy as cp
from implicit.gpu.als import AlternatingLeastSquares

import pickle 

from loaders import lastfm, movielens
import utils

class BaseModel:
	def __init__(self, R):
		"""
			R is an m x n binary ratings matrix where m is the number
			of users and n is the number of items.
		"""
		self.R = R
		self.n, self.m = R.shape

	def predict_ratings(self, d):
		"""
			Returns a prediction ratings matrix of the same shape as self.R.
			Input d is the embedding dimension.
		"""
		raise NotImplementedError()

	def get_R(self):
		return self.R
		
class PCA(BaseModel):
	def __init__(self, R):
		super().__init__(R)
		U, S, Vh = svd(self.R, full_matrices=False)
		self.Vh = Vh

	def predict_ratings(self, d, use_diagonal=True):
		assert d <= len(self.Vh)
		P = self.Vh[:d].T @ self.Vh[:d]
		if not use_diagonal:
			P -= np.diag(np.diag(P))
		return self.R @ P

	def get_specialization(self, d):
		V = self.Vh[:d].T
		item_sim = V @ V.T
		# V = np.linalg.inv(np.diag(np.linalg.norm(V, axis=1))) @ V
		# item_sim = np.exp(V @ V.T)
		# item_sim = item_sim @ np.linalg.inv(np.diag(np.sum(item_sim, axis=0)))
		return np.diag(item_sim)

class NormalizedPCA(PCA):
	def __init__(self, R):
		self.R = R @ np.linalg.inv(np.diag(np.linalg.norm(R, axis = 0)))
		U, S, Vh = svd(self.R, full_matrices=False)
		self.Vh = Vh

class ItemWeightedPCA(BaseModel):
	def __init__(self, R, dataset_name):
		super().__init__(R)
		self.name = dataset_name

	def save_projection_matrix(self, P, d, gamma, file_suffix=""):
		pred_mtxs_dict = {}
		try: 
			with open("pickles/%s%s.pickle" % (self.name, file_suffix), "rb") as pickleFile:
				pred_mtxs_dict = pickle.load(pickleFile)
		except:
			pass

		if d not in pred_mtxs_dict:
			pred_mtxs_dict[d] = {}
		pred_mtxs_dict[d][gamma] = P
		with open("pickles/%s%s.pickle" % (self.name, file_suffix), "wb") as pickleFile:
			pickle.dump(pred_mtxs_dict, pickleFile)	

	def _get_P(self, R, d, gamma=-1, recompute=True, save=False, max_iters=-1, verbose=False, file_suffix="_dense"):
		if not recompute:
			with open("pickles/%s%s.pickle" % (self.name, file_suffix), "rb") as pickleFile:
				pred_mtxs_dict = pickle.load(pickleFile)
				P = pred_mtxs_dict[d][gamma]
				return P

		# save a placeholder to confirm that the saving mechanism works
		if save:
			self.save_projection_matrix(np.zeros((self.m, self.m)), d, gamma, file_suffix)

		proj_mtx = cp.Variable((self.m, self.m), PSD=True) #eigenval of proj_mtx are >= 0
		
		constraints = [
			np.eye(self.m) - proj_mtx >> 0, #eigenvalues of proj_mtx are <= 1
			cp.trace(proj_mtx) == d,
			# cp.trace(cov_mtx @ proj_mtx) >= accuracy_bound * vanilla_obj_value
		]
		
		norms = np.linalg.norm(R, axis=0)
		print("Minimum norm: {}".format(np.min(norms)))
		weights = norms**gamma
		weight_mtx = np.diag(weights)

		coefs = weight_mtx.T @ R.T @ R
		prob = cp.Problem(cp.Minimize(-cp.trace(coefs @ proj_mtx)),
						  constraints)
		if max_iters > 0:
			prob.solve(solver=cp.SCS, 
				max_iters=max_iters,
				time_limit_secs=60 * 60, 
				verbose=verbose)
				# use_indirect=False,
				# mkl=False, 
				# gpu=False)
		else:
			prob.solve(solver=cp.SCS, 
				verbose=verbose,
				time_limit_secs=60*60)
				# use_indirect=False,
				# mkl=False, 
				# gpu=False)
			
		if prob.status in ["infeasible", "unbounded"]:
			print("SDP Problem is {}".format(prob.status))
			return np.zeros((self.m, self.m))
		
		proj = proj_mtx.value
		if save:
			self.save_projection_matrix(proj, d, gamma, file_suffix)
		return proj

	def predict_ratings(self, d, gamma=-1, max_iters=-1, recompute=False, save=False, use_diagonal=True, file_suffix="_dense"):
		P = self._get_P(self.R, d, gamma,
			recompute=recompute,
			save=save, 
			max_iters=max_iters,
			verbose=True,
			file_suffix=file_suffix)

		actual_rank = np.sum(np.linalg.eigvals(P) > 0.1)
		if abs(actual_rank - d) > 0:
			print("Expected rank: %i Actual rank: %i" % (d, actual_rank))

		## enforce the rank constraint
		V, _, _ = svd(P) #singular vec sorted in non-increasing order
		V = V[:, :d]
		P = V @ V.T

		if not use_diagonal:
			P -= np.diag(np.diag(P))
			
		# P = v @ np.diag(w) @ v.T

		return self.R @ P

	def get_specialization(self, d, gamma=-1, file_suffix="_dense"):
		P = self._get_P(self.R, d, gamma, recompute=False, save=False, file_suffix=file_suffix)
		V, _, _ = svd(P)
		V = V[:, :d]
		item_sim = V @ V.T
		# V = np.linalg.inv(np.diag(np.linalg.norm(V, axis=1))) @ V
		# item_sim = np.exp(V @ V.T)
		# item_sim = item_sim @ np.linalg.inv(np.diag(np.sum(item_sim, axis=0)))
		return np.diag(item_sim)

class LightGCN(BaseModel):
	def __init__(self, R, dataset_name):
		super().__init__(R)
		self.dataset_name = dataset_name
		self.filename = "pickles/lgn-predictions-{}.pickle".format(dataset_name)

	def predict_ratings(self, d, use_diagonal=True):
		'''
			use_diagonal is not relevant for non-projection based models
		'''
		with open(self.filename, "rb") as pickleFile:
			predictions = pickle.load(pickleFile)
			return predictions[d]["ratings"]

	def get_specialization(self, d):
		with open(self.filename, "rb") as pickleFile:
			predictions = pickle.load(pickleFile)
			V = predictions[d]["item embeddings"].cpu().detach().numpy()
		assert V.shape == (self.m, d)

		# V = np.linalg.inv(np.diag(np.linalg.norm(V, axis=1))) @ V
		item_sim = np.exp(V @ V.T)
		item_sim = item_sim @ np.linalg.inv(np.diag(np.sum(item_sim, axis=0)))
		return np.diag(item_sim)

class MF(BaseModel):
	def __init__(self, R, dataset_name, alpha=1.0):
		super().__init__(R)
		self.dataset_name = dataset_name
		self.alpha = alpha
		self.saved_results = {}

	def predict_ratings(self, d, use_diagonal=True):
		als = AlternatingLeastSquares(factors=d, iterations=50)

		rescaled_R = self.R + (self.R > 0)
		als.fit(csr_matrix(rescaled_R))
		U = als.user_factors.to_numpy()
		V = als.item_factors.to_numpy()
		self.saved_results[d] = (U, V)

		return U @ V.T

	def get_specialization(self, d):
		if d not in self.saved_results:
			self.predict_ratings(d)
		_, V = self.saved_results[d]
		assert V.shape == (self.m, d)

		# V = np.linalg.inv(np.diag(np.linalg.norm(V, axis=1))) @ V
		item_sim = np.exp(V @ V.T)
		item_sim = item_sim @ np.linalg.inv(np.diag(np.sum(item_sim, axis=0)))
		return np.diag(item_sim)



## TODO: implement weighted matrix factorization in PyTorch
if __name__ == "__main__":
	R = None
	if sys.argv[1] == "lastfm":
		obj = lastfm.lastfm()
		# setting min_ratings = 10 minimizes the number of users with no test interactions
		obj.filter(min_users = 10, min_ratings = 10)
		R = obj.get_P()
		R[R != 0] = 1
	elif sys.argv[1] == "lastfm-explicit":
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
	elif sys.argv[1] == "movielens":
		R = movielens.movielens(min_ratings = 1, min_users = 5, binary = True).get_X()
		R[R != 0] = 1

		rng = np.random.default_rng(seed=2020)
		R = R[:, rng.choice(a=R.shape[1], size=1200, replace=False)]


	print(R.shape)

	R_train, R_val, R_test = utils.split_data(R, 
		val_ratio = 0.1, 
		test_ratio = 0.2)

	print("Minimum number of users: %f" % np.min(np.sum(R_train + R_val, axis=0)))
	model = ItemWeightedPCA(R_train + R_val, sys.argv[1])

	taskId = int(os.getenv('SLURM_ARRAY_TASK_ID'))

	ds = utils.get_ds(R)

	# gammas = [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1]
	# d = 32
	gamma = -1
	# gamma = gammas[taskId]

	d = ds[taskId]
	# print("d = %i" % d)

	# ds = [2**i for i in range(1, 11)]
	# for d in ds:
	# d = 2
	yhat = model.predict_ratings(d = d, gamma=gamma, recompute=True, save=True, file_suffix="_dense")

	# predict_partial = partial(model.predict_ratings, 
	# 	max_iters=200,
	# 	recompute=True,
	# 	save=False)
	# with multiprocessing.Pool() as pool:
	# 	pool.map(predict_partial, ds)

	# yhat_sorted = utils.get_prediction_matrix(yhat, R_train + R_val, R_test)

	# k = 10
	# recall, precis = utils.RecallPrecision(R_test, yhat_sorted, k)
	# mrr = utils.MRR(yhat_sorted, k) / len(R_test)
	# ndcg = utils.NDCG(R_test, yhat_sorted, k) / len(R_test)

	# print("Recall: {:0.2f}\t Precision: {:0.2f}\t MRR: {:0.2f}\t NDCG: {:0.2f}".format(
	# 	recall,
	# 	precis,
	# 	mrr,
	# 	ndcg))