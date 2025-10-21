SEED=42
BATCH_SIZE=100
DATASET_NAME="CREMA-D"
CLASS_NUM=6
INPUT_MODALITY="video"
TRAINED_MODEL_FILE="CREMA-D_classNum6_20251021_091649_epoch3_0.7660_seed42_dropout0.3.pth"   # prepretrainで学習した重みファイル名
HIDDEN_DIM=768

LOG_FILE="test_results/prepretest/${INPUT_MODALITY}/${DATASET_NAME}_${CLASS_NUM}_$(date +%Y%m%d_%H%M%S).log"

python -u prepretest.py \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --dataset_name $DATASET_NAME \
    --class_num $CLASS_NUM \
    --input_modality $INPUT_MODALITY \
    --trained_model_file $TRAINED_MODEL_FILE \
    --hidden_dim $HIDDEN_DIM \
    2>&1 | tee "$LOG_FILE"