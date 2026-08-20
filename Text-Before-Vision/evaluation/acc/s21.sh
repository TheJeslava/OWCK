. ./load_exp.sh
echo $STEP_PATH
sleep 2


trap 'echo -e "\n\n[停止] 检测到 Ctrl+C，正在退出循环..."; exit 0' SIGINT

echo "开始运行 vLLM 服务..."

while true; do
    python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port ${PORT} \
        --model ~/tmp/${EXP_NAME}/global_step_${STEP}/actor/huggingface \
        --served-model-name ${EXP_NAME}_step_${STEP} \
        --limit-mm-per-prompt '{"image": 32}' \
        --gpu-memory-utilization 0.35  --mm-processor-cache-gb 0 --tensor-parallel-size 2
        # --gpu-memory-utilization 0.8  --mm-processor-cache-gb 0
        # --gpu-memory-utilization 0.15 

    echo "----------------------------------------"
    echo "程序已退出 (Exit Code: $?)。"
    echo "将在 3 秒后自动重启... (按 Ctrl+C 立刻终止)"
    echo "----------------------------------------"
    
    sleep 3
done
