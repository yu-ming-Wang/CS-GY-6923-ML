#!/bin/bash
#SBATCH -p nvidia            # Specifies the GPU partition
#SBATCH --gres=gpu:1        # Requests 1 GPU
#SBATCH -n 1                 # Number of tasks
#SBATCH -c 10                # Number of CPUs per task
#SBATCH --mem=400G            # Memory per node
#SBATCH -t 30:00:00          # Wall time (1 day)

# module load python/3.8 cuda/11.0  # Example modules: Python and CUDA
export CUDA_HOME=/share/apps/NYUAD5/cuda/11.8.0
export CUDNN_HOME=/archive/apps/work/apps/dalma-share-apps_backup/apps/NYUAD5/schrodinger/2021/mmshare-v5.5/lib/Linux-x86_64
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_HOME:$LD_LIBRARY_PATH


# Run your GPU-enabled Python script
# /home/jl10897/.conda/envs/deep_personality/bin/python script/run_exp.py -c config/unified_frame_images/MOSE_10_swin_transformer.yaml
python script/run_exp.py -c config/unified_face_images/MOSE_swin_transformer_Alia_face.yaml


