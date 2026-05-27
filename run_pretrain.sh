#!/bin/bash
set -e

# Load modules (same order as run_rank.sh).
module load gcc/14.2.0
module load openmpi/5.0.10-gcc
module load cuda/12.8.0
module load cudnn/9.19.0

source /gs/bs/hp190122/ux07012/.venv/llm_lecture/bin/activate
cd ~/develop/LLM_lecture/Megatron-LM

# Make gcc 14's libstdc++ visible.
GCC_LIBDIR=$(dirname $(g++ -print-file-name=libstdc++.so.6))
export LD_LIBRARY_PATH="$GCC_LIBDIR:$LD_LIBRARY_PATH"

# pip NCCL.
NCCL_DIR=$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")
export LD_LIBRARY_PATH="$NCCL_DIR/lib:$LD_LIBRARY_PATH"

# HuggingFace cache on group storage (home quota is tight).
export HF_HOME=/gs/bs/hp190122/ux07012/.cache/huggingface

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(head -n 1 "$SGE_JOB_SPOOL_DIR/pe_hostfile" | cut -d " " -f 1)
export MASTER_PORT=29508
export PYTHONPATH=$PWD

# Tiny GPT for sanity check.
MODEL_ARGS=(
  --num-layers 4
  --hidden-size 256
  --num-attention-heads 4
  --seq-length 512
  --max-position-embeddings 512
  --position-embedding-type learned_absolute
)

TRAIN_ARGS=(
  --micro-batch-size 2
  --global-batch-size 4
  --train-iters 20
  --lr 1e-4
  --min-lr 1e-5
  --lr-decay-style cosine
  --weight-decay 0.1
  --clip-grad 1.0
  --bf16
)

PARALLEL_ARGS=(
  --tensor-model-parallel-size 2
  --pipeline-model-parallel-size 1
)

DATA_ARGS=(
  --data-path processed_data_text_document
  --tokenizer-type GPT2BPETokenizer
  --vocab-file tokenizers/gpt2/vocab.json
  --merge-file tokenizers/gpt2/merges.txt
  --split 90,5,5
)

LOG_ARGS=(
  --log-interval 1
  --save-interval 1000
  --eval-interval 1000
  --eval-iters 0
  --save ckpt_pretrain
  --load ckpt_pretrain
)

mpirun -np 2 -npernode=1 \
  -x CUDA_DEVICE_MAX_CONNECTIONS -x MASTER_ADDR -x MASTER_PORT \
  -x PYTHONPATH -x LD_LIBRARY_PATH -x HF_HOME \
  bash -c 'export RANK=$OMPI_COMM_WORLD_RANK \
                  WORLD_SIZE=$OMPI_COMM_WORLD_SIZE \
                  LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK; \
           exec python pretrain_gpt.py \
             '"${MODEL_ARGS[*]}"' \
             '"${TRAIN_ARGS[*]}"' \
             '"${PARALLEL_ARGS[*]}"' \
             '"${DATA_ARGS[*]}"' \
             '"${LOG_ARGS[*]}"
