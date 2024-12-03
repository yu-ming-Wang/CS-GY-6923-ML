#!/bin/bash
#SBATCH --time=3:30:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1 
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err



# singularity exec --overlay /scratch/jl10897/overlay-50G-4000K.ext3 /scratch/jl10897/MOSErun.sif /bin/bash -c "source activate deep_personality && python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir '/archive/jl10897/test' --output-dir '/archive/jl10897/test_overlay_data'"
# singularity exec --overlay /scratch/jl10897/overlay-50G-4000K.ext3 /scratch/jl10897/MOSErun.sif python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir "/archive/jl10897/test" --output-dir "/archive/jl10897/test_overlay_data"


# singularity exec --overlay /scratch/jl10897/overlay-50G-4000K.ext3 /scratch/jl10897/MOSErun.sif conda run -n deep_personality python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir '/archive/jl10897/test' --output-dir '/archive/jl10897/test_overlay_data'
# run-overlay -o /scratch/jl10897/overlay-50G-4000K.ext3 -f /scratch/jl10897/file.txt

# singularity exec --overlay /scratch/jl10897/overlay-50G-4000K.ext3 /scratch/jl10897/MOSErun.sif /bin/bash -c "
#     source /opt/conda/etc/profile.d/conda.sh && \
#     conda activate deep_personality && \
#     python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py \
#     --video-dir '/archive/jl10897/test' \
#     --output-dir '/archive/jl10897/test_overlay_data'
# "

overlay_ext3=/scratch/jl10897/overlay-50G-4000K.ext3

# singularity \
#     exec --overlay $overlay_ext3:ro \
#     /scratch/jl10897/MOSErun.sif \
#     /bin/bash -c "source /home/jl10897/.bashrc \
#                 conda init bash \
#                 conda activate /home/jl10897/.conda/envs/deep_personality; \
#                 /home/jl10897/.conda/envs/deep_personality/bin/python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir '/archive/jl10897/test' --output-dir '/archive/jl10897/test_overlay_data' "

# singularity \
#     exec --overlay $overlay_ext3:ro \
#     /scratch/jl10897/MOSErun.sif \
#     /home/jl10897/.conda/envs/deep_personality/bin/python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir '/archive/jl10897/test' --output-dir '/archive/jl10897/test_overlay_data' 


# singularity \
#     exec --overlay $overlay_ext3:ro \
#     /scratch/jl10897/MOSErun.sif \
#     /home/jl10897/.conda/envs/deep_personality/bin/python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir '/scratch/jl10897/DeepPersonality-main/datasets/MOSE/test' --output-dir '/archive/jl10897/test_overlay_data' 
#!/bin/bash
#SBATCH --time=3:30:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1 
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err



# singularity exec --overlay /scratch/jl10897/overlay-50G-4000K.ext3 /scratch/jl10897/MOSErun.sif /bin/bash -c "source activate deep_personality && python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir '/archive/jl10897/test' --output-dir '/archive/jl10897/test_overlay_data'"
# singularity exec --overlay /scratch/jl10897/overlay-50G-4000K.ext3 /scratch/jl10897/MOSErun.sif python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir "/archive/jl10897/test" --output-dir "/archive/jl10897/test_overlay_data"


# singularity exec --overlay /scratch/jl10897/overlay-50G-4000K.ext3 /scratch/jl10897/MOSErun.sif conda run -n deep_personality python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir '/archive/jl10897/test' --output-dir '/archive/jl10897/test_overlay_data'
# run-overlay -o /scratch/jl10897/overlay-50G-4000K.ext3 -f /scratch/jl10897/file.txt

# singularity exec --overlay /scratch/jl10897/overlay-50G-4000K.ext3 /scratch/jl10897/MOSErun.sif /bin/bash -c "
#     source /opt/conda/etc/profile.d/conda.sh && \
#     conda activate deep_personality && \
#     python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py \
#     --video-dir '/archive/jl10897/test' \
#     --output-dir '/archive/jl10897/test_overlay_data'
# "

overlay_ext3=/scratch/jl10897/overlay-50G-4000K.ext3

# singularity \
#     exec --overlay $overlay_ext3:ro \
#     /scratch/jl10897/MOSErun.sif \
#     /bin/bash -c "source /home/jl10897/.bashrc \
#                 conda init bash \
#                 conda activate /home/jl10897/.conda/envs/deep_personality; \
#                 /home/jl10897/.conda/envs/deep_personality/bin/python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir '/archive/jl10897/test' --output-dir '/archive/jl10897/test_overlay_data' "

# singularity \
#     exec --overlay $overlay_ext3:ro \
#     /scratch/jl10897/MOSErun.sif \
#     /home/jl10897/.conda/envs/deep_personality/bin/python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir '/archive/jl10897/test' --output-dir '/archive/jl10897/test_overlay_data' 


# singularity \
#     exec --overlay $overlay_ext3:ro \
#     --bind /archive/jl10897:/archive \
#     --bind /scratch/jl10897:/scratch \
#     /scratch/jl10897/MOSErun.sif \
#     /home/jl10897/.conda/envs/deep_personality/bin/python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir '/scratch/jl10897/DeepPersonality-main/datasets/MOSE/test' --output-dir '/archive/jl10897/test_overlay_data' 
    
    
singularity \
    exec --overlay $overlay_ext3:ro \
    --bind /archive/jl10897:/archive,/scratch/jl10897:/scratch \
    /scratch/jl10897/MOSErun.sif \
    /home/jl10897/.conda/envs/deep_personality/bin/python /scratch/jl10897/DeepPersonality-main/dpcv/data/utils/use_video_to_image.py --video-dir '/scratch/jl10897/DeepPersonality-main/datasets/MOSE/test' --output-dir '/archive/jl10897/test_overlay_data'

    
    