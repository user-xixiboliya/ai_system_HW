#!/bin/bash
# train.sh: 调用训练脚本，传入默认参数

# 默认参数设置
DATA_DIR="../data"
BATCH_SIZE=64
BATCH_NUM=""         # 如果为空，则不传入该参数，可在代码中处理 None 的情况
HIDDEN_SIZE=1024
LR=2e-5
EPOCHS=10
DEVICE=0

# 如果需要传入 batch_num 参数，可以取消注释以下行并赋予具体值
# BATCH_NUM=YOUR_VALUE

# 打印参数信息（可选）
echo "Training with parameters:"
echo "  data_dir    = ${DATA_DIR}"
echo "  batch_size  = ${BATCH_SIZE}"
echo "  hidden_size = ${HIDDEN_SIZE}"
echo "  lr          = ${LR}"
echo "  epochs      = ${EPOCHS}"
echo "  device      = ${DEVICE}"

# 运行训练脚本
# 如果 BATCH_NUM 参数为空，则可以不传入；否则传入参数
if [ -z "$BATCH_NUM" ]; then
    python NNI_modified.py --data_dir "${DATA_DIR}" \
                    --batch_size "${BATCH_SIZE}" \
                    --hidden_size "${HIDDEN_SIZE}" \
                    --lr "${LR}" \
                    --epochs "${EPOCHS}" \
                    --device "${DEVICE}"
else
    python NNI_modified.py --data_dir "${DATA_DIR}" \
                    --batch_size "${BATCH_SIZE}" \
                    --batch_num "${BATCH_NUM}" \
                    --hidden_size "${HIDDEN_SIZE}" \
                    --lr "${LR}" \
                    --epochs "${EPOCHS}" \
                    --device "${DEVICE}"
fi