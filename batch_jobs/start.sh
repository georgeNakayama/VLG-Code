#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --partition=preempt
#SBATCH -A marlowe-m000051
#SBATCH --job-name=resume
#SBATCH --output=train-multi-%x.%j.out
#SBATCH --error=train-multi-%x.%j.err

export CUDA_HOME=/cm/shared/apps/nvhpc/24.7/Linux_x86_64/24.7/cuda/12.5
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export WORKDIR=/scratch/m000051/garment_gang/AIpparel-Code
# Number of nodes
export NNODES=1
# Number of GPUs per node
export GPUS_PER_NODE=8

mamba activate llama

cd $WORKDIR

PYTHONPATH=$WORKDIR:$WORKDIR/src torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE training_scripts/train_aipparel_llama3.py \
    --config-name train_v2 \
    run_name="long-train-single-node" \
    project="vlg-train"
