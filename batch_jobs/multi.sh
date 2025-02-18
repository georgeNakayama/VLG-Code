#!/bin/bash -l
#SBATCH --nodes=2                      # Request 2 nodes
#SBATCH --ntasks-per-node=1             # One task per node (torchrun will handle GPUs)
#SBATCH --gres=gpu:8                    # Request 8 GPUs per node
#SBATCH --cpus-per-task=64              # CPU cores per task
#SBATCH --mem=512G                      # Memory per node
#SBATCH --time=12:00:00                 # Max time
#SBATCH --partition=preempt             # Partition
#SBATCH -A marlowe-m000051              # Account
#SBATCH --job-name=train-multi
#SBATCH --output=train-multi.out
#SBATCH --error=train-multi.err

# Load environment
source /scratch/m000051/miniforge3/bin/activate
mamba activate llama

# Set up distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # First node as master
export MASTER_PORT=29500  # Port for communication
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=8
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))

module load nvhpc
export PIP_CACHE_DIR="/scratch/m000051/.cache/pip/"
export TORCH_EXTENSIONS_DIR=/scratch/m000051/.cache/torch_extensions
export TRITON_CACHE_DIR=/scratch/m000051/.cache/triton_cache
export CUDA_CACHE_PATH=/scratch/m000051/.cache/cuda
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export NINJA_CACHE=/scratch/m000051/.cache/ninja
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/scratch/m000051/.cache/huggingface/
export HUGGINGFACE_TOKEN=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
export CXX=$(which g++)
export CC=$(which gcc)

# Print SLURM setup
echo "Master node: $MASTER_ADDR"
echo "Number of nodes: $NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs (World Size): $WORLD_SIZE"

# Setup stuff
export WORKDIR=/scratch/m000051/garment_gang/AIpparel-Code
export PYTHONPATH=$WORKDIR:$WORKDIR/src 

cd $WORKDIR

# Run torch distributed training
srun bash -c 'torchrun \
    --nnodes=$NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    training_scripts/train_aipparel_llama3.py \
    --config-name train_v2 \
    run_name="test-sbatch" \
    project="vlg-train"'

