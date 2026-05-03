# 4*90GB

DATASET_PATH="raw_data/MCSD1/ms_swift/train_sample_Thinking_sft_all.jsonl"
VAL_DATASET_PATH="raw_data/MCSD1/ms_swift/val_sample_Thinking_sft_all.jsonl"
OUTPUT_DIR="shared_output/mcsd"
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --tuner_type lora \
    --dataset "$DATASET_PATH" \
    --val_dataset "$VAL_DATASET_PATH" \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --max_length 20000 \
    --output_dir "$OUTPUT_DIR/sft_all_reasoning" \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --report_to wandb