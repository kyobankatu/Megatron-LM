#!/bin/bash
set -e

module load openmpi/5.0.10-gcc
source /gs/bs/hp190122/ux07012/.venv/llm_lecture/bin/activate
cd ~/develop/LLM_lecture/Megatron-LM

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(head -n 1 "$SGE_JOB_SPOOL_DIR/pe_hostfile" | cut -d " " -f 1)
export MASTER_PORT=29508
export PYTHONPATH=$PWD

mpirun -np 2 -npernode=1 \
  -x CUDA_DEVICE_MAX_CONNECTIONS -x MASTER_ADDR -x MASTER_PORT -x PYTHONPATH \
  bash -c 'export RANK=$OMPI_COMM_WORLD_RANK \
                  WORLD_SIZE=$OMPI_COMM_WORLD_SIZE \
                  LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK; \
           exec python examples/run_simple_mcore_train_loop.py'
