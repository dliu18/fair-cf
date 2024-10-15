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
    assert len(sys.argv) == 2
    _, dataset_name = sys.argv
    print(dataset_name)
    
    X = 0
    if dataset_name == "movielens":
        X = get_movielens()
    elif dataset_name == "lastfm":
        X = get_lastfm()
    assert len(X) > 0
    print(X.shape)
    
    d = X.shape[1]
    rs = np.arange(1, d, 15)
    alphas = np.arange(0, 1, 0.05)
    
    filename = "pickles/max_pred_filter_all_rs_{}_cosine.pickle".format(dataset_name)
    proj_matrices = {}
    with open(filename, "wb") as pickleFile:
        description = '''
        Projection matrices from the max prediction algorithm with random rantings set to zero. Alpha is the fraction of entries set to zero. Index 1 contains the projection matrices indexed by r then by alpha
        '''
        pickle.dump((description, X, proj_matrices), pickleFile)
    
    for r in tqdm(rs):
        proj_matrices[r] = {}
        rng = np.random.default_rng(2**10)
        nonzero_idxs = np.transpose(np.nonzero(X))
        num_nonzero = len(nonzero_idxs)
        rng.shuffle(nonzero_idxs)

        for alpha in alphas:
            lam = utils.fair_pca_max_pred
            X_filtered = X.copy()
            for idx in nonzero_idxs[:int(alpha * num_nonzero)]:
                X_filtered[idx[0], idx[1]] = 0
            proj_matrices[r][alpha] = lam(X_filtered, d, r, cosine = True)

            with open(filename, "wb") as pickleFile:
                description = '''
                Projection matrices from the max prediction algorithm with random rantings set to zero. Alpha is the fraction of entries set to zero. Index 1 contains the projection matrices indexed by alpha
                '''
                pickle.dump((description, X, proj_matrices), pickleFile)
    
    