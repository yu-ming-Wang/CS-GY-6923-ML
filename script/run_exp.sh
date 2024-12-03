#!/bin/bash
#SBATCH -p nvidia            # Specifies the GPU partition
#SBATCH --gres=gpu:1         # Requests 1 GPU
#SBATCH -n 1                 # Number of tasks
#SBATCH -c 10                # Number of CPUs per task
#SBATCH --mem=32G            # Memory per node
#SBATCH -t 24:00:00          # Wall time (1 day)

# module load python/3.8 cuda/11.0  # Example modules: Python and CUDA
# conda activate deep_personality

# Run your GPU-enabled Python script
python script/run_exp.py -c config/unified_frame_images/10_swin_transformer.yaml
