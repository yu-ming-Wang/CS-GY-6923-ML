#!/bin/bash
#SBATCH --time=3:30:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1 
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err


/home/jl10897/.conda/envs/deep_personality/bin/python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/zipping_test.py

    
    