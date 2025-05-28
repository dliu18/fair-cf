# Reproducing "When Collaborative Filtering is not Collaborative: Unfairness of PCA for Recommendations"

Authors: [David Liu](https://dliu18.github.io/), [Jackie Baek](https://jwbaek.github.io/), [Tina Eliassi-Rad](https://eliassi.org/)

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

All of the models are implemented in `models.py`. It is best to generate the Item-Weighted PCA projection matrices in advance and save them to intermediary files. For the paper, we generate three batches of results, differing in how they define $r$ and $\gamma$. The three batches are:
* _sparse_: fixes $\gamma=-1$ and runs Item-Weighted PCA for $r \in {2, 4, 8, ... , 1024}$.
* _dense_: also fixes $\gamma=-1$ but does a more fine grained sweep of $r$.
* _sweep\_gamma_: fixes $r=32$ and sweeps $\gamma \in {-2, -1.9, ... , 0}$.

All of the batches can be executed with `sbatch sbatch.sh`. Before each call, update line 4 with the correct number of jobs, as documented in the same file. Also, in `models.py`, uncomment the corresponding code block in the main function. Ensure the file_suffix parameter (second parameter to `models.py` is set to "\_sparse", "\_dense", or "\_sweep\_gamma").

### Training LightGCN

Execute `sweep_dimensions.sh` in `LightGCN/code` via: 
```
./sweep_dimensions.sh movielens
```
The script will run the LightGCN model for each value of $r \in {2, 4, 8, ... , 1024}$. The output embeddings and predictions for all models will be stored in a single pickle file. Recall that we only run LightGCN for the movielens dataset as LightGCN requires binary interaction data.

The implementation of LightGCN was forked from [gusye1234/LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch/tree/master/code)

### Instruction to reproduce results in paper

The results figures from the paper can all be reproduced via the `main.py` script. Below, we specify the modifications to `main.py` needed for each figure: 
* Figure 2: Uncomment the call to `_specialization` in the main function (see "Uncomment depending on the figure"). Also uncomment `rs = utils.get_ds(R)`. Execute:
```
python main.py lastfm-explicit -1 dense
python main.py movielens -1 dense
``` 

* Figure 3: Uncomment the call to `_performance_by_popularity`. Note that setting `out_sample=False` ensures that the performance metrics are _in-sample_. Also uncomment `rs = utils.get_ds(R)`. Execute:
```
python main.py lastfm-explicit -1 dense
python main.py movielens -1 dense
``` 

* Figure 4: Uncomment `_aggregate_performance` and ensure `rs = [2**i for i in range(2, 11)]`. Execute:
```
python main.py lastfm-explicit -1 sparse
python main.py movielens -1 sparse
``` 

* Figure 6: Already reproduced following the steps for Figure 4. 

* Figure 7: In `_aggregate_performance`, uncomment the addition of "Weighted MF" and "LightGCN" to the `model_list`. Ensure that all four metric names are included in `metrics`. Comment out the PCA baseline. In the main function, ensure `rs = [2**i for i in range(2, 11)]`. Execute: 
```
python main.py lastfm-explicit -1 sparse
python main.py movielens -1 sparse
``` 

* Figure 8: In the main function, uncomment `_performance_by_gamma`. Execute:
```
python main.py lastfm-explicit -1 sweep_gamma
python main.py movielens -1 sweep_gamma
``` 

For any of the above figures, also comment the final two lines of `main.py` to save the figure values to a csv file.

## Miscellaneous

Utilize the script `python check_pickle.py file_suffix` to view the contents of an Item-Weighted PCA pickle file. Specify the pickle file with a file_suffix (the "file_suffix" passed to `models.py`). 

The teaser figure (Figure 1), can be reproduced via `Teaser.ipynb`.

The empirical validation in Figure 5 can be reproduced via `Bernoulli matrix eigenvalue scaling.ipynb`.



