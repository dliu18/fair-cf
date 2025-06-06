#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=fair-pca
#SBATCH --array=0-9%25
#SBATCH --ntasks=1
#SBATCH --output=logs/out_array_%A_%a.out
#SBATCH --error=logs/err_array_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G

source ~/.bashrc
mamba activate fair-ranking
work
cd fair-cf
python models.py lastfm-explicit _sparse


# Num of jobs to instantiate 

# _sparse
# lastfm-explicit --array=0-9%25
# movielens --array=0-9%25

# _dense
# lastfm-explicit --array=0-23%25
# movielens --array=0-21%25

# _sweep_gamma 
# lastfm-explicit --array=0-20%25
# movielens --array=0-20%25