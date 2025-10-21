# パラメータ設定
SEED=42
LR=1e-4
EPOCHS=50
BATCH_SIZE=32
# MOSIなら2か7
# CREMA-Dなら6
DATASET_NAME="MOSI"     # 固定
CLASS_NUM=2             # 固定
INPUT_MODALITY="video"
HIDDEN_DIM=768
DROPOUT_RATE=0.3
PRETRAINED_MODEL_FILE="CREMA-D_classNum6_20251021_091649_epoch8_0.7788_seed42_dropout0.3.pth"
PATIENCE=5


# 動的ログファイル名生成
LOG_FILE="logs/prepretrain/${INPUT_MODALITY}/${DATASET_NAME}_${CLASS_NUM}_$(date +%Y%m%d_%H%M%S)_${DROPOUT_RATE}.log"

python -u prepretrain.py \
    --seed $SEED \
    --lr $LR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset_name $DATASET_NAME \
    --class_num $CLASS_NUM \
    --input_modality  $INPUT_MODALITY \
    --hidden_dim $HIDDEN_DIM \
    --dropout_rate $DROPOUT_RATE \
    --pretrained_model_file $PRETRAINED_MODEL_FILE \
    --patience $PATIENCE \
    2>&1 | tee "$LOG_FILE"