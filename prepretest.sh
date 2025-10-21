SEED=42
BATCH_SIZE=100
DATASET_NAME="MOSI"
CLASS_NUM=2
INPUT_MODALITY="video"
TRAINED_MODEL_FILE="MOSI_classNum2_20251022_075740_epoch0_0.6026_seed42_dropout0.3.pth"   # prepretrainで学習した重みファイル名
HIDDEN_DIM=768


# 動的ログファイル名生成
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