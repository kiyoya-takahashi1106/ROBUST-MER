# パラメータ設定
SEED=42
LR=1e-4
DROPOUT_RATE=0.3
EPOCHS=100
BATCH_SIZE=100
DATASET_NAME="CREMA-D"
PREPRETRAINED_MODEL_FILE="CREMA-D"
CLASS_NUM=6
INPUT_MODALITY="audio"
PRETRAINED_MODEL_FILE="CREMA-D_epoch20_20251016_080303_0.7204_seed42.pth"
HIDDEN_DIM=768
WEIGHT_SIM=2.0
WEIGHT_DIFF=50.0
WEIGHT_RECON=0.3
WEIGHT_TASK=0.15
WEIGHT_DISCRIMINATOR=1.0
PATIENCE=5


# 動的ログファイル名生成
LOG_FILE="logs/pretrain/${INPUT_MODALITY}/${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"

python -u pretrain.py \
    --seed $SEED \
    --lr $LR \
    --dropout_rate $DROPOUT_RATE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset_name $DATASET_NAME \
    --prepretrained_model_name $PREPRETRAINED_MODEL_FILE \
    --class_num $CLASS_NUM \
    --input_modality  $INPUT_MODALITY \
    --pretrained_model_file $PRETRAINED_MODEL_FILE \
    --hidden_dim $HIDDEN_DIM \
    --weight_sim $WEIGHT_SIM \
    --weight_diff $WEIGHT_DIFF \
    --weight_recon $WEIGHT_RECON \
    --weight_task $WEIGHT_TASK \
    --weight_discriminator $WEIGHT_DISCRIMINATOR \
    --patience $PATIENCE \
    2>&1 | tee "$LOG_FILE"