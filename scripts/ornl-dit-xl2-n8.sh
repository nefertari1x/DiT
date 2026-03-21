#!/bin/bash
#SBATCH -A lrn075
#SBATCH -o imagenet_shf-vit-b.o%J
#SBATCH -t 02:00:00
#SBATCH -N 8
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
srun -N 8 -n 64 --ntasks-per-node 8 python ./train_feat_bf16_cp_bz_scale.py \
    --model DiT-XL/2 \
    --features-path /work/c30778/dataset/dit_feat_fix/train 