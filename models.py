import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import eigsh

import multiprocessing
from functools import partial
import sys 
import os 

import cvxpy as cp

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

class NormalizedPCA(PCA):
	def __init__(self, R):
		self.R = R @ np.linalg.inv(np.diag(np.linalg.norm(R, axis = 0)))
		U, S, Vh = svd(self.R, full_matrices=False)
		self.Vh = Vh

class ItemWeightedPCA(BaseModel):
	def __init__(self, R, dataset_name):
		super().__init__(R)
		self.name = dataset_name

	def _get_P(self, R, d, accuracy_bound = 0, cosine = False, max_iters = -1, verbose = False):
	    '''
	        Setting cosine to True modifies the objective to approximate the cosine similarity between the predicted and 
	        actual values.
	    '''
	    proj_mtx = cp.Variable((self.m, self.m), PSD=True) #eigenval of proj_mtx are >= 0
	    
	    cov_mtx = R.T @ R
	    w, _ = eigsh(cov_mtx, which = "LM", k = d)
	    vanilla_obj_value = np.sum(w)
	    
	    constraints = [
	        np.eye(self.m) - proj_mtx >> 0, #eigenvalues of proj_mtx are <= 1
	        cp.trace(proj_mtx) == d,
	        # cp.trace(cov_mtx @ proj_mtx) >= accuracy_bound * vanilla_obj_value
	    ]
	    
	    weight_mtx = np.zeros((self.n, self.m))
	    for j in range(self.m):
	        norm = np.linalg.norm(R[:, j])
	        if cosine: # if R is binary, "cosine" does not make a difference
	            norm = np.linalg.norm(R[:, j] != 0)
	        for i in range(self.n):
	            if R[i, j] > 0:
	                weight_mtx[i, j] = 1 / norm
	            elif R[i, j] < 0:
	                weight_mtx[i, j] = -1 / norm
	#             if cosine:
	#                 weight_mtx[i, j] *= abs(X[i, j])
	    prob = cp.Problem(cp.Minimize(-cp.trace(weight_mtx.T @ (R @ proj_mtx))),
	                      constraints)
	    if max_iters > 0:
	        prob.solve(solver=cp.SCS, 
	        	max_iters=max_iters,
	        	time_limit_secs=60 * 30, 
	        	verbose=verbose)
	        	# use_indirect=False,
	        	# mkl=False, 
	        	# gpu=False)
	    else:
	        prob.solve(solver=cp.SCS, 
	        	verbose=verbose,
	        	time_limit_secs=60*30)
	        	# use_indirect=False,
	        	# mkl=False, 
	        	# gpu=False)
	        
	    if prob.status in ["infeasible", "unbounded"]:
	        print("SDP Problem is {}".format(prob.status))
	        return np.zeros((self.m, self.m))
	    
	    proj = proj_mtx.value
	    return proj

	def predict_ratings(self, d, max_iters=-1, recompute=False, save=False, use_diagonal=True):
		P = np.zeros((self.m, self.m))
		try:
			if recompute:
				raise NotImplementedError

			with open("pickles/%s.pickle" % self.name, "rb") as pickleFile:
				pred_mtxs_dict = pickle.load(pickleFile)
				P = pred_mtxs_dict[d]
		except:
			P = self._get_P(self.R, d, max_iters=max_iters, cosine=True, verbose=True)
			if save:
				pred_mtxs_dict = {}
				try: 
					with open("pickles/%s.pickle" % self.name, "rb") as pickleFile:
						pred_mtxs_dict = pickle.load(pickleFile)
				except:
					pass

				pred_mtxs_dict[d] = P
				with open("pickles/%s.pickle" % self.name, "wb") as pickleFile:
					pickle.dump(pred_mtxs_dict, pickleFile)

		if not use_diagonal:
			P -= np.diag(np.diag(P))

		return self.R @ P

class LightGCN(BaseModel):
	def __init__(self, R):
		super().__init__(R)

class MF(BaseModel):
	def __init__(self, R):
		super().__init__(R)


## TODO: implement weighted matrix factorization in PyTorch
if __name__ == "__main__":
	R = None
	if sys.argv[1] == "LastFM":
		obj = lastfm.lastfm()
		# setting min_ratings = 10 minimizes the number of users with no test interactions
		obj.filter(min_users = 10, min_ratings = 10)
		R = obj.get_P()
		R[R != 0] = 1
	elif sys.argv[1] == "MovieLens":
		R = movielens.movielens(min_ratings = 1, min_users = 5, binary = True).get_X()
		R[R != 0] = 1

		rng = np.random.default_rng(seed=2020)
		R = R[:, rng.choice(a=R.shape[1], size=1200, replace=False)]


	R_train, R_val, R_test = utils.split_data(R, 
		val_ratio = 0.1, 
		test_ratio = 0.2)

	model = ItemWeightedPCA(R_train + R_val, sys.argv[1])

	taskId = int(os.getenv('SLURM_ARRAY_TASK_ID'))
	d = 2**taskId
	print("d = %i" % d)
	# ds = [2**i for i in range(1, 8)]
	# for d in ds:
		
	yhat = model.predict_ratings(d = d, recompute=True, save=True)

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