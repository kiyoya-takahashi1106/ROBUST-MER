SEED=42
LR=1e-4
DROPOUT_RATE=0.3
EPOCHS=100
BATCH_SIZE=100
DATASET_NAME="CREMA-D"
CLASS_NUM=6
HIDDEN_DIM=768

AUDIO_PRETRAINED_MODEL_FILE="CREMA-D_classNum6_20251020_212809_epoch8_0.7201_seed42_dropout0.3.pth"
VIDEO_PRETRAINED_MODEL_FILE="CREMA-D_classNum6_20251021_091649_epoch8_0.7788_seed42_dropout0.3.pth"

# 動的ログファイル名生成
LOG_FILE="logs/compare_cremaD/${DATASET_NAME}_${CLASS_NUM}_$(date +%Y%m%d_%H%M%S).log"

python -u compare_cremaD_train.py \
    --seed $SEED \
    --lr $LR \
    --dropout_rate $DROPOUT_RATE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset_name $DATASET_NAME \
    --class_num $CLASS_NUM \
    --hidden_dim $HIDDEN_DIM \
    --audio_pretrained_model_file $AUDIO_PRETRAINED_MODEL_FILE \
    --video_pretrained_model_file $VIDEO_PRETRAINED_MODEL_FILE \
    2>&1 | tee "$LOG_FILE"