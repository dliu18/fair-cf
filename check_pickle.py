import pickle
import sys
import numpy as np

filename = sys.argv[1]

with open("pickles/%s.pickle" % filename, "rb") as pickleFile:
	pred_mtxs = pickle.load(pickleFile)

for d in pred_mtxs:
	for gamma in pred_mtxs[d]:
		print(gamma)
		P = pred_mtxs[d][gamma]
		actual_rank = np.sum(np.linalg.eigvals(P) > 0.01)
		ortho_rank = np.sum(np.linalg.eigvals(P) > 0.99)
		print(d, gamma, P.shape, not np.all(P == 0), np.sum(P) > 0, actual_rank, ortho_rank)