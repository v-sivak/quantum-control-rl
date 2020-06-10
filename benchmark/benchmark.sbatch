#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=9
#SBATCH --mem=40gb
#SBATCH --constraint=cascadelake
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=benchmark
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=henry.c.liu@yale.edu

source ~/.bashrc
module restore gkp-rl
conda activate gkp-rl

# Compare single displacement generation (i.e. initialization time)
python benchmark_algo.py --num 1
python benchmark_algo.py --num 1 --dataset

# A typical "in-memory" batch size
python benchmark_algo.py --num 100
python benchmark_algo.py --num 100 --dataset

# "Largest" (conservatively) batch size that fits in 32GB
python benchmark_algo.py --num 0
python benchmark_algo.py --num 0 --dataset --batchsize 0

# Tests a larger than memory batch with different batchsizes
python benchmark_algo.py --num 500 --dataset --batchsize 100
python benchmark_algo.py --num 500 --dataset --batchsize 0
