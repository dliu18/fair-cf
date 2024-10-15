import sys
from movielens import movielens
import lastfm
import numpy as np
import utils
from tqdm import tqdm
import pickle

def get_movielens():
    movielens_obj = movielens(top_k_users = 2000, top_k_items = 1000, binary = False)
    idx_to_genres = movielens_obj.get_genres()
    X = movielens_obj.get_X()
    assert X.shape[1] == len(idx_to_genres)

    genres_to_idx = {}
    for idx in idx_to_genres:
        for genre in idx_to_genres[idx]:
            if genre not in genres_to_idx:
                genres_to_idx[genre] = []
            genres_to_idx[genre].append(idx)

    column_sample = []
    for genre in genres_to_idx:
        idxs = np.array(genres_to_idx[genre])
        top_idx = np.argsort(np.sum(X[:, idxs] != 0, axis = 0))[-30:]
        new_column_sample = column_sample.copy()
        for idx in idxs[top_idx]:
            if idx not in column_sample:
                new_column_sample.append(idx)
        column_sample = new_column_sample

    X = X[:, column_sample]

    X_top_movies = np.zeros(X.shape)
    for i in range(len(X)):
        top_movie_idxs = np.argsort(X[i])[-30:]
        X_top_movies[i, top_movie_idxs] = X[i, top_movie_idxs]
    
    X_binary = np.zeros(X.shape)
    X_binary[X_top_movies == 1] = -2
    X_binary[X_top_movies == 2] = -1
    X_binary[X_top_movies == 3] = 1
    X_binary[X_top_movies == 4] = 2
    X_binary[X_top_movies == 5] = 3

    return X_binary / 100

def get_lastfm():
    lastfm_obj = lastfm.lastfm(data_dir = "data/lastfm/")
    P = lastfm_obj.filter(min_users = 50, min_ratings = 20)

    P = np.linalg.inv(np.diag(np.sum(P, axis=1))) @ P
    assert np.allclose(np.sum(P, axis=1), np.ones(len(P)))
    return P

if __name__ == "__main__":
    assert len(sys.argv) == 3
    _, dataset_name, train_test = sys.argv
    print(dataset_name)
    print(train_test)
    
    X = 0
    if dataset_name == "movielens":
        X = get_movielens()
    elif dataset_name == "lastfm":
        X = get_lastfm()
    assert len(X) > 0
    print(X.shape)
    
    idxs = np.arange(len(X))
    if train_test == "train":
        rng = np.random.default_rng(2**10)
        train_idxs = rng.choice(len(X), size = int(0.7 * len(X)), replace = False)
        idxs = train_idxs.copy()
#         test_idxs = []
#         for idx in range(len(X)):
#             if idx not in train_idxs:
#                 test_idxs.append(idx)
#         idxs = np.array(test_idxs)
    
    d = X.shape[1]
    rs = np.arange(1, d)
    
    filename = "pickles/max_pred_{}_{}_cosine.pickle".format(dataset_name, train_test)
    proj_matrices = {}
    with open(filename, "wb") as pickleFile:
        description = '''
        Projection matrices from the max prediction algorithm. Index 1 contains the idxs of the matrix used for training.
        Index 2 contains the projection matrices indexed by r
        '''
        pickle.dump((description, idxs, proj_matrices), pickleFile)
        
    for r in tqdm(rs):
        lam = utils.fair_pca_max_pred
        proj_matrices[r] = lam(X[idxs], d, r, cosine = False)
        
        with open(filename, "wb") as pickleFile:
            description = '''
            Projection matrices from the max prediction algorithm. Index 1 contains the idxs of the matrix used for training.
            Index 2 contains the projection matrices indexed by r
            '''
            pickle.dump((description, idxs, proj_matrices), pickleFile)
    
    