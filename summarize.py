from loaders import lastfm, movielens
import numpy as np

def summarize(name, R):
	print(name)
	print(f"n: {len(R)} m: {R.shape[1]}")
	print(f"entires: {np.sum(R > 0)}")
	print(f"min value: {np.min(R)} max value: {np.max(R)} avg. non-zero value: {np.mean(R[R>0])}")
	print("\n")

# LastFM
obj = lastfm.lastfm()
R_orig = obj.get_P()
summarize("LastFM original", R_orig)

obj.filter(min_users = 10, min_ratings = 10)
R = obj.get_P()
R = np.linalg.inv(np.diag(np.sum(R, axis=1))) @ R
summarize("LastFM filtered", R)

# movielens
R_orig = movielens.movielens(binary = True).get_X()
summarize("movielens original", R_orig)

R = movielens.movielens(min_ratings = 1, min_users = 5, binary = True).get_X()
R[R != 0] = 1
rng = np.random.default_rng(seed=2020)
R = R[:, rng.choice(a=R.shape[1], size=1200, replace=False)]
summarize("movielens filtered", R)
