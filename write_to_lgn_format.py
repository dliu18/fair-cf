'''
Output the lastfm dataset used in the fair PCA paper to the format that LightGCN expects
Usage: python write_to_lgn_format.py 
'''

from loaders import lastfm, movielens
import numpy as np
import utils
import sys

def write_to_file(ratings, dataset_name, split):
    with open("LightGCN/data/{}/{}.txt".format(dataset_name, split), "w") as txtfile:
        all_item_idxs = np.array(range(ratings.shape[1]))
        for u_idx in range(len(ratings)):
            item_idxs = all_item_idxs[ratings[u_idx] > 0]
            
            row = str(u_idx)
            for item_idx in item_idxs:
                row += (" " + str(item_idx))
            row += "\n"
            
            txtfile.write(row)
            
if __name__ == "__main__":
    dataset_name = sys.argv[1]
    R = None
    assert dataset_name in ["lastfm", "movielens"]

    if dataset_name == "lastfm":
        obj = lastfm.lastfm()
        # setting min_ratings = 10 minimizes the number of users with no test interactions
        obj.filter(min_users = 10, min_ratings = 10)
        R = obj.get_P()
        R[R != 0] = 1
    elif dataset_name == "movielens":
        R = movielens.movielens(min_ratings = 1, min_users = 5, binary = True).get_X()
        R[R != 0] = 1

        rng = np.random.default_rng(seed=2020)
        R = R[:, rng.choice(a=R.shape[1], size=1200, replace=False)]

    R_train, R_val, R_test = utils.split_data(R, 
        val_ratio = 0.1, 
        test_ratio = 0.2)

    write_to_file(R_train + R_val, dataset_name, "train")
    write_to_file(R_test, dataset_name, "test")