#!/bin/bash
#SBATCH -A lrn075
#SBATCH -o exfeat_IN1K.o%J
#SBATCH -t 02:00:00
#SBATCH -N 4
#SBATCH -p batch
#SBATCH --mail-user=zhangsuiyu657@gmail.com
#SBATCH --mail-type=END

# --- Environment Setup for ROCm/MIOPEN on your HPC ---
export MIOPEN_DISABLE_CACHE=1 
export MIOPEN_CUSTOM_CACHE_DIR='pwd' 
export HOME="/tmp/srun"

# --- Load Required Modules ---
# This part is specific to your HPC environment (e.g., Frontier, Crusher)
echo "Loading modules..."
module load PrgEnv-gnu
module load gcc-native/12.3
module load rocm/6.4.0
echo "Modules loaded."


echo "Launching distributed training..."
srun -N 4 -n 32 --ntasks-per-node 8 python ./extract_features.py \
    --data-path /lustre/orion/nro108/world-shared/enzhi/dataset/imagenet \
    --features-path /lustre/orion/nro108/world-shared/enzhi/dataset/ \
    --vae-path /lustre/orion/nro108/world-shared/enzhi/DiT/model/vae_ema