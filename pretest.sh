SEED=42
BATCH_SIZE=100
DATASET_NAME="CREMA-D"
PREPRETRAINED_DATASET="CREMA-D"
PREPRETRAINED_CLASSNUM=6
CLASS_NUM=6
INPUT_MODALITY="video"
TRAINED_MODEL_FILE="CREMA-D_classNum6_20251021_120618_epoch6_0.2895_0.9792_seed42_dropout0.3.pth"   # prepretrainで学習した重みファイル名
HIDDEN_DIM=768

LOG_FILE="test_results/pretest/${INPUT_MODALITY}/${PREPRETRAINED_DATASET}_${PREPRETRAINED_CLASSNUM}_$(date +%Y%m%d_%H%M%S).log"

python -u pretest.py \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --dataset_name $DATASET_NAME \
    --prepretrained_dataset $PREPRETRAINED_DATASET \
    --prepretrained_classnum $PREPRETRAINED_CLASSNUM \
    --class_num $CLASS_NUM \
    --input_modality $INPUT_MODALITY \
    --trained_model_file $TRAINED_MODEL_FILE \
    --hidden_dim $HIDDEN_DIM \
    2>&1 | tee "$LOG_FILE"