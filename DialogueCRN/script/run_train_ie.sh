#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

WORK_DIR="./DialogueCRN"


EXP_NO="dialoguecrn_v1"
DATASET="iemocap"
echo "${EXP_NO}, ${DATASET}"


DATA_DIR="${WORK_DIR}/data/${DATASET}/IEMOCAP_features.pkl"
OUT_DIR="${WORK_DIR}/outputs/${DATASET}/${EXP_NO}"
MODEL_DIR="${WORK_DIR}/outputs/iemocap/dialoguecrn_v1/dialoguecrn_22.pkl"

LOG_PATH="${WORK_DIR}/logs/${DATASET}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

G="0 0.5 1 2"
S="0 1 2 3 4 5 6"

for g in ${G[@]}
do
    for ss in ${S[@]}
    do
        for sp in ${S[@]}
        do
        echo "gamma:${g}, step_s: ${ss}, step_p: ${sp}"
        python -u ${WORK_DIR}/code/run_train_ie.py   \
            --status train  --feature_type text  --data_dir ${DATA_DIR}  --output_dir ${OUT_DIR}  --load_model_state_dir ${MODEL_DIR} \
            --gamma $g  --step_s ${ss}  --step_p ${sp}  --lr 0.0001  --l2 0.0002  --dropout 0.2  --base_layer 2  --valid_rate 0.0 \
        >> ${LOG_PATH}/${EXP_NO}.out 2>&1

        done
    done
done