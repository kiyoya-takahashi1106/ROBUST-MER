SEED=42
BATCH_SIZE=100
DATASET_NAME="CREMA-D"
CLASS_NUM=6
HIDDEN_DIM=768
TRAINED_MODEL_FILE="CREMA-D_classNum6_20251021_135339_epoch19_0.8045_seed42_dropout0.3.pth"

LOG_FILE="test_results/test/${DATASET_NAME}_${CLASS_NUM}_$(date +%Y%m%d_%H%M%S).log"

python -u test.py \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --dataset_name $DATASET_NAME \
    --class_num $CLASS_NUM \
    --hidden_dim $HIDDEN_DIM \
    --trained_model_file $TRAINED_MODEL_FILE \
    2>&1 | tee "$LOG_FILE"