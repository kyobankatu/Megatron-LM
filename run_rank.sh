#!/bin/bash
set -e

source ~/.bashrc
conda activate llm_lecture
cd ~/develop/LLM_lecture/Megatron-LM

export CUDA_DEVICE_MAX_CONNECTIONS=1

NODE_RANK=$1
MASTER=$2
PORT=$3

torchrun \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER} \
  --master_port=${PORT} \
  examples/run_simple_mcore_train_loop.py
