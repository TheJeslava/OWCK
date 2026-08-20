. ./load_exp.sh
echo $STEP_PATH
sleep 2

# export CUDA_VISIBLE_DEVICES=5
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --model ./tmp/${EXP_NAME}/global_step_${STEP}/actor/huggingface \
    --served-model-name ${EXP_NAME}_step_${STEP} \
    --limit-mm-per-prompt '{"image": 12}' \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.45
