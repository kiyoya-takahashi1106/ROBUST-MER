SEED=42
LR=1e-4
DROPOUT_RATE=0.3
EPOCHS=100
BATCH_SIZE=120
DATASET_NAME="MOSI"
CLASS_NUM=2
HIDDEN_DIM=768
AUDIO_PRETRAINED_MODEL_FILE="MOSI_classNum2_epoch6_20251017_123601_0.1743_0.7931_seed42_dropout0.3.pth"
TEXT_PRETRAINED_MODEL_FILE="MOSI_classNum2_epoch1_20251016_134023_0.7817_seed42_dropout0.3.pth"
VIDEO_PRETRAINED_MODEL_FILE="MOSI_classNum2_epoch10_20251016_134922_0.6594_seed42_dropout0.3.pth"


# 動的ログファイル名生成
LOG_FILE="logs/train/${DATASET_NAME}_${CLASS_NUM}_$(date +%Y%m%d_%H%M%S).log"

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
    --text_pretrained_model_file $TEXT_PRETRAINED_MODEL_FILE \
    --video_pretrained_model_file $VIDEO_PRETRAINED_MODEL_FILE \
    2>&1 | tee "$LOG_FILE"