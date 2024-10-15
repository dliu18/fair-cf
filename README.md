Code to Reproduce KDD 2024 Submission "When Collaborative Filtering is not Collaborative: Unfairness of PCA for Recommendations"

## General Notes

* A conda yml file specifying the computing environment can be found in environment.yml.
* The directories `figs` and `pickles` are created as empty directories that will be populated by the scripts.
    

## Data
The datasets can be accessed from the GroupLens website. For LastFM, download the following [zip file](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip) and unzip the files into `data/lastfm`. For MovieLens, download the following [zip file](https://files.grouplens.org/datasets/movielens/ml-1m.zip) and place the unzipped folder (`ml-1m`) into `data/`.
    
## Instruction to reproduce results

The first step is to run our algorithm Item-Preference PCA on the LastFM and Movielens datasets for all values of r. To do so, execute:
```
    python max_prediction.py lastfm all
    python max_prediction.py movielens all
    
    python max_prediction_filter.py lastfm
    python max_prediction_filter.py movielens
```
Each line above requires approximately 3 hours of runtime on a machine with Intel Xeon E5-2690 CPUs, 2.6GHz, 30 MB of cache.

The latter two lines are for the robustness results. 

Now, the Figures can be reproduced by executing the Jupyter notebooks.
To reproduce 
* Figure 1, execute `Teaser.ipynb`
* Figures 2, 3, 4, 5, 7 execute `Item-Preference PCA Evaluation.ipynb`
* Figure 6 execute `Bernoulli matrix eigenvalue scaling.ipynb`

## Miscellaneous

We found that multiplying the Movielens matrix by a constant factor of 1/100 greatly improved runtime. Multiplying by a constant factor does not affect the solution for Item-Preference PCA given that the objective function is linear.
