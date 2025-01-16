import pickle
import sys
import numpy as np

filename = sys.argv[1]

with open("pickles/%s.pickle" % filename, "rb") as pickleFile:
	pred_mtxs = pickle.load(pickleFile)

# new_pred_mtxs = {}
# for d in pred_mtxs:
# 	new_pred_mtxs[d] = {-1: pred_mtxs[d]}

# with open("pickles/%s_revised.pickle" % filename, "wb") as pickleFile:
# 	pickle.dump(new_pred_mtxs, pickleFile)

for d in pred_mtxs:
	for gamma in pred_mtxs[d]:
		P = pred_mtxs[d][gamma]
		actual_rank = np.sum(np.linalg.eigvals(P) > 0.01)
		ortho_rank = np.sum(np.linalg.eigvals(P) > 0.99)
		print(d, gamma, P.shape, not np.all(P == 0), np.sum(P) > 0, actual_rank, ortho_rank)