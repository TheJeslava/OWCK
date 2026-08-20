. ./load_exp.sh
echo $STEP_PATH
sleep 2


mkdir -p results/${EXP_NAME}_step_${STEP}
# IP=100.99.198.192
IP=127.0.0.1
python eval_multi_xlrsbench2.py \
    --model_name ${EXP_NAME}_step_${STEP} \
    --api_key None \
    --api_url http://${IP}:${PORT}/v1 \
    --xlrsbench_path /path/to/converted_xlrs_data \
    --save_path results/${EXP_NAME}_step_${STEP} \
    --eval_model_name ${EXP_NAME}_step_${STEP} \
    --num_workers 16
