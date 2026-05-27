#!/bin/bash
set -e

# Load modules. Order matters: gcc/14 must override system libstdc++,
# and cuda/12.8 must come AFTER openmpi (which auto-pulls cuda/13.1.1).
module load gcc/14.2.0
module load openmpi/5.0.10-gcc
module load cuda/12.8.0
module load cudnn/9.19.0

source /gs/bs/hp190122/ux07012/.venv/llm_lecture/bin/activate
cd ~/develop/LLM_lecture/Megatron-LM

# Make gcc 14's libstdc++ (with CXXABI_1.3.15) visible at runtime.
GCC_LIBDIR=$(dirname $(g++ -print-file-name=libstdc++.so.6))
export LD_LIBRARY_PATH="$GCC_LIBDIR:$LD_LIBRARY_PATH"

# Make pip-installed NCCL headers/libs visible (needed by TE at runtime).
NCCL_DIR=$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")
export LD_LIBRARY_PATH="$NCCL_DIR/lib:$LD_LIBRARY_PATH"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=$(head -n 1 "$SGE_JOB_SPOOL_DIR/pe_hostfile" | cut -d " " -f 1)
export MASTER_PORT=29508
export PYTHONPATH=$PWD

mpirun -np 2 -npernode=1 \
  -x CUDA_DEVICE_MAX_CONNECTIONS -x MASTER_ADDR -x MASTER_PORT -x PYTHONPATH \
  -x LD_LIBRARY_PATH \
  bash -c 'export RANK=$OMPI_COMM_WORLD_RANK \
                  WORLD_SIZE=$OMPI_COMM_WORLD_SIZE \
                  LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK; \
           exec python examples/run_simple_mcore_train_loop.py'
