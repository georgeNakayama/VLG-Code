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
#SBATCH --output=train-multi-%x.%j.out
#SBATCH --error=train-multi-%x.%j.err

# Load environment
source /scratch/m000051/miniforge3/bin/activate
mamba activate llama

# Set up distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # First node as master
export MASTER_PORT=29500  # Port for communication
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=8
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export CUDA_HOME=/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/cuda/12.5
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH


# Print SLURM setup
echo "Master node: $MASTER_ADDR"
echo "Number of nodes: $NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs (World Size): $WORLD_SIZE"

# Setup stuff
export WORKDIR=/scratch/m000051/garment_gang/AIpparel-Code
export PYTHONPATH=$WORKDIR:$WORKDIR/src 

cd $WORKDIR

export BASE_DIR="/scratch/m000051/garment_gang/vlg-train/long-train-2-nodes/2025-02-22"
mv $BASE_DIR/16-39-01/ckpt_9000/global_step9001/* $BASE_DIR/16-39-02/ckpt_9000/global_step9001/
mv $BASE_DIR/16-39-03/ckpt_9000/global_step9001/* $BASE_DIR/16-39-02/ckpt_9000/global_step9001/

# TODO(george): Fill this in.
export RESUME_DIR="/scratch/m000051/garment_gang/vlg-train/long-train-2-nodes/2025-02-22/16-39-02/ckpt_9000"
export RUN_ID="kg7c9746"
export RUN_LOCAL_PATH="/scratch/m000051/garment_gang/wandb/run-20250221_020637-kg7c9746"

# Run torch distributed training
srun bash -c 'torchrun \
    --nnodes=$NNODES \
    --node_rank=$SLURM_NODEID \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    training_scripts/train_aipparel_llama3.py \
    --config-name train_v2 \
    resume=$RESUME_DIR \
    run_name="long-train-2-nodes" \
    run_id=$RUN_ID \
    run_local_path=$RUN_LOCAL_PATH \
    grad_accumulation_steps=5 \
    project="vlg-train"'

