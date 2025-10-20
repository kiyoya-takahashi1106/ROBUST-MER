# パラメータ設定
SEED=42
LR=1e-4
EPOCHS=50
BATCH_SIZE=36
# MOSIなら2か7
# CREMA-Dなら6
DATASET_NAME="MOSI"
CLASS_NUM=2
INPUT_MODALITY="audio"
HIDDEN_DIM=768
DROPOUT_RATE=0.3
PATIENCE=5
CREMAD_WEIGHT_FILE=CREMA-D_classNum6_epoch5_20251016_130433_0.7500_seed42_dropout0.3.pth    


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
    --patience $PATIENCE \
    --cremad_weight_file $CREMAD_WEIGHT_FILE \
    2>&1 | tee "$LOG_FILE"