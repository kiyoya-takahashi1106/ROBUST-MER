# パラメータ設定
SEED=42
LR=1e-4
DROPOUT_RATE=0.5
EPOCHS=100
BATCH_SIZE=20
DATASET_NAME="CREMA-D"
CLASS_NUM=6
INPUT_MODALITY="audio"
INPUT_DIM_AUDIO=74
INPUT_DIM_VIDEO=47
HIDDEN_DIM=128
BERT_MODEL="bert-base-uncased"

# 動的ログファイル名生成
LOG_FILE="logs/pretrain/${INPUT_MODALITY}/seed${SEED}_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"

python -u pretrain.py \
    --seed $SEED \
    --lr $LR \
    --dropout_rate $DROPOUT_RATE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset_name $DATASET_NAME \
    --class_num $CLASS_NUM \
    --input_modality  $INPUT_MODALITY \
    --input_dim_audio $INPUT_DIM_AUDIO \
    --input_dim_video $INPUT_DIM_VIDEO \
    --hidden_dim $HIDDEN_DIM \
    --bert_model_name $BERT_MODEL \
    2>&1 | tee "$LOG_FILE"