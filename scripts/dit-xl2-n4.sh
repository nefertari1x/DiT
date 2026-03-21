#!/bin/sh
#------ qsub option --------#
#PBS -q lg
#PBS -l select=4:ngpus=4:mpiprocs=4
#PBS -l walltime=24:00:00
#PBS -W group_list=c30636
#PBS -j oe

# 1. 加载模块
module load gcc ompi

# 2. 获取主节点 IP
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500

# 3. [核心修复] 自动探测正确的以太网卡名称
# 逻辑：列出所有 IPv4 地址 -> 排除 lo, docker, 169.254 (link-local) -> 取第一个剩下的网卡名
# 这一步是为了避开导致报错的 169.254.3.1
export DETECTED_IFNAME=$(ip -o -4 addr show | awk '!/^[0-9]+: lo|docker|169\.254/ {print $2}' | head -n 1)

# 如果探测失败，回退到排除法（尝试排除 usb0，因为 169.254 常出现在 usb0）
if [ -z "$DETECTED_IFNAME" ]; then
    echo "Warning: Auto-detection failed, using fallback exclusion."
    export GLOO_SOCKET_IFNAME=^lo,docker0,usb0,virbr0
    export NCCL_SOCKET_IFNAME=^lo,docker0,usb0,virbr0
else
    echo "Auto-detected valid interface: $DETECTED_IFNAME"
    export GLOO_SOCKET_IFNAME=$DETECTED_IFNAME
    export NCCL_SOCKET_IFNAME=$DETECTED_IFNAME
fi

# 4. 数据传输配置 (保持不变，强制使用 IB)
export NCCL_IB_HCA=mlx5
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=INFO

echo "Master Node: $MASTER_ADDR"
echo "Socket Interface: $GLOO_SOCKET_IFNAME"

# >>> conda initialize >>>
__conda_setup="$('/home/c30746/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/c30746/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/c30746/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/c30746/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

cd /work/c30636/DiT
conda activate DiT

# 5. 启动命令
# 务必通过 -x 传递 DETECTED_IFNAME 相关的变量
mpirun -np 4 \
    --map-by ppr:1:node \
    --bind-to none \
    -x MASTER_ADDR \
    -x MASTER_PORT \
    -x GLOO_SOCKET_IFNAME \
    -x NCCL_SOCKET_IFNAME \
    -x NCCL_IB_HCA \
    -x NCCL_IB_DISABLE \
    -x NCCL_P2P_DISABLE \
    -x NCCL_DEBUG \
    -x PATH \
    torchrun \
    --nnodes=4 \
    --nproc_per_node=4 \
    --rdzv_id=$PBS_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./train_feat_bf16_cp_bz_snr.py --model DiT-XL/2 --features-path /work/c30778/dataset/dit_feat_fix/train 
    # --resume ./results/035-DiT-XL-2/checkpoints/0040000.pt