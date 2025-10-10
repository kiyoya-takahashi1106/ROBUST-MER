# パラメータ設定
SEED=42
LR=1e-4
EPOCHS=100
BATCH_SIZE=16
DATASET_NAME="CREMA-D"
CLASS_NUM=6
INPUT_MODALITY="audio"
HIDDEN_DIM=768


# 動的ログファイル名生成
LOG_FILE="logs/prepretrain/${INPUT_MODALITY}/${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"

python -u prepretrain.py \
    --seed $SEED \
    --lr $LR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset_name $DATASET_NAME \
    --class_num $CLASS_NUM \
    --input_modality  $INPUT_MODALITY \
    --hidden_dim $HIDDEN_DIM \
    2>&1 | tee "$LOG_FILE"