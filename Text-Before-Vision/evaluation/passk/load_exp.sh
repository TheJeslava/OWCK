export XDG_CACHE_HOME=$HOME/._cache
export TRITON_CACHE_DIR=$HOME/._cache/triton
export TORCHINDUCTOR_CACHE_DIR=$HOME/._cache/torch/inductor
export VLLM_CACHE_ROOT=$HOME/._cache/vllm
export PYTORCH_KERNEL_CACHE_PATH=$HOME/._cache/torch/kernels


STEP=32
# PORT=8080
PORT=8000
# EXP_NAME=debug_1027_math_coldstart

STEP_PATH=/path/to/${PROJECT_NAME}/${EXP_NAME}/global_step_${STEP}
