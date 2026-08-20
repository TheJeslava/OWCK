. ./load_exp.sh
echo $STEP_PATH
sleep 2
# exit 0
# copy config files
mkdir -p ./tmp/${EXP_NAME}/global_step_${STEP}/actor/huggingface
cp -r ${STEP_PATH}/actor/huggingface/* ./tmp/${EXP_NAME}/global_step_${STEP}/actor/huggingface
echo "Converting to hf format"
python /path/to/DeepEyes/scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path ${STEP_PATH}/actor/huggingface \
    --local_dir ${STEP_PATH}/actor \
    --target_dir ./tmp/${EXP_NAME}/global_step_${STEP}/actor/huggingface
