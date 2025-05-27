**["When Collaborative Filtering is not Collaborative: Unfairness of PCA for Recommendations"](https://arxiv.org/abs/2310.09687)**

[David Liu](https://dliu18.github.io/), Jackie Baek, Tina Eliassi-Rad

Published FAccT'25


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

Three different calls to `sbatch.sh`

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



