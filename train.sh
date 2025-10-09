# パラメータ設定
SEED=42
LR=1e-4
DROPOUT_RATE=0.5
EPOCHS=100
BATCH_SIZE=130
DATASET_NAME="CREMA-D"
CLASS_NUM=6
HIDDEN_DIM=768
AUDIO_PRETRAINED_MODEL_FILE=".pth"
VIDEO_PRETRAINED_MODEL_FILE=".pth"


# 動的ログファイル名生成
LOG_FILE="logs/pretrain/${DATASET_NAME}/$(date +%Y%m%d_%H%M%S).log"

python -u pretrain.py \
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