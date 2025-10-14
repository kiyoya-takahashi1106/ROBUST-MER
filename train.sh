# パラメータ設定
SEED=42
LR=1e-4
DROPOUT_RATE=0.3
EPOCHS=100
BATCH_SIZE=80
DATASET_NAME="MOSI"
CLASS_NUM=1
HIDDEN_DIM=768
AUDIO_PRETRAINED_MODEL_FILE="epoch46_0.0405_0.9763_seed42.pth"
VIDEO_PRETRAINED_MODEL_FILE="epoch90_0.0413_0.7456_seed42.pth"


# 動的ログファイル名生成
LOG_FILE="logs/train/${DATASET_NAME}/$(date +%Y%m%d_%H%M%S).log"

python -u train.py \
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