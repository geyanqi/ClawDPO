#!/bin/sh
set -eu

# 基于 ms-swift 官方 full DPO recipe：
# https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/dpo/full.sh
# 数据路径是唯一位置参数，其余稳定训练参数留在脚本内。
if [ "$#" -ne 1 ] || [ ! -e "$1" ]; then
    echo "usage: $0 <dataset-path>" >&2
    exit 2
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

exec swift rlhf \
    --rlhf_type dpo \
    --model "${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}" \
    --tuner_type full \
    --dataset "$1" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir "${OUTPUT_DIR:-output}" \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --rpo_alpha 0.1 \
    --padding_free true
