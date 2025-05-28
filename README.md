# Reproducing "When Collaborative Filtering is not Collaborative: Unfairness of PCA for Recommendations"

Authors: [David Liu](https://dliu18.github.io/), Jackie Baek, Tina Eliassi-Rad

Published at FAccT'25

[ArXiv](https://arxiv.org/abs/2310.09687)

## General Notes

* A conda yml file specifying the computing environment can be found in environment.yml.
* The directories `figs` and `pickles` are created as empty directories that will be populated by the scripts.
    

## Data
The datasets can be accessed from the GroupLens website. For LastFM, download the following [zip file](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip) and unzip the files into `data/lastfm`. For MovieLens, download the following [zip file](https://files.grouplens.org/datasets/movielens/ml-1m.zip) and place the unzipped folder (`ml-1m`) into `data/`.

To create the train/test datasets for LightGCN, execute:
```
python write_to_lgn_format.py lastfm
python write_to_lgn_format.py movielens
```

## Execution
   
### Training Item-Weighted PCA

All of the models are implemented in `models.py`. It is best to generate the Item-Weighted PCA projection matrices in advance and save them to intermediary files. For the paper, we generate three batches of results, differing in how they define $d$ and $\gamma$. The three batches are:
* _sparse_: fixes $\gamma=-1$ and runs Item-Weighted PCA for $d \in \{2, 4, 8, ... , 1024\}$.
* _dense_: also fixes $\gamma=-1$ but does a more fine grained sweep of $d$.
* _sweep gamma_: fixes $d=32$ and sweeps $\gamma \in \{-2, -1.9, ... , 0\}$.

All of the batches can be executed with `sbatch sbatch.sh`. Before each call, update line 4 with the correct number of jobs, as documented in the same file. Also, in `models.py`, uncomment the corresponding code block in the main function.  

### Training LightGCN

Execute `sweep_dimensions.sh` in `LightGCN/code`.

```
./sweep_dimensions.sh movielens
```

The script already calls `read_models.py`.


### Instruction to reproduce results in paper

Three different calls to `main.py`


## Miscellaneous

Utilize the script `python check_pickle.py file_suffix` to view the contents of an Item-Weighted PCA pickle file. Specify the pickle file with a file_suffix (the "file_suffix" passed to `models.py`). 

The teaser figure (Figure 1), can be reproduced via `Teaser.ipynb`.

The empirical validation in Figure 5 can be reproduced via `Bernoulli matrix eigenvalue scaling.ipynb`.



