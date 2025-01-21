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
python models_copy.py lastfm-explicit

#LastFM array size = 24
#movielens array size = 22 