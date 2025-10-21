SEED=42
BATCH_SIZE=100
DATASET_NAME="CREMA-D"
CLASS_NUM=6
HIDDEN_DIM=768
DROPOUT_RATE=0.3
AUDIO_PRETRAINED_MODEL_FILE="test.pth"
VIDEO_PRETRAINED_MODEL_FILE="test.pth"
TRAINED_MODEL_FILE="CREMA-D_classNum6_20251021_180950_epoch0_0.8590_seed42_dropout0.3.pth"

LOG_FILE="test_results/compare_cremaD/${DATASET_NAME}_${CLASS_NUM}_$(date +%Y%m%d_%H%M%S).log"

python -u compare_cremaD_test.py \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --dataset_name $DATASET_NAME \
    --class_num $CLASS_NUM \
    --hidden_dim $HIDDEN_DIM \
    --dropout_rate $DROPOUT_RATE \
    --audio_pretrained_model_file $AUDIO_PRETRAINED_MODEL_FILE \
    --video_pretrained_model_file $VIDEO_PRETRAINED_MODEL_FILE \
    --trained_model_file $TRAINED_MODEL_FILE \
    2>&1 | tee "$LOG_FILE"