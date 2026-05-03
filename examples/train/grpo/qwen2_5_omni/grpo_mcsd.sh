# 4 * 90GiB
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

DATASET_PATH="/scratch/p311104/ms_home/ms-swift/jsonl_data_train/mcsd/grpo_train_zh.jsonl"
VAL_DATASET_PATH="/scratch/p311104/ms_home/ms-swift/jsonl_data_train/mcsd/grpo_val_zh_100.jsonl"
CHECKPOINT_PATH="saved_out/mcsd/sft-reasoning-all/checkpoint-1000-merged"
RESUME_PATH="saved_out/mcsd/grpo_reasoning_genrm1/v7-20260502-100411/checkpoint-700"
RM_PATH="saved_out/my_genrm_3b_mcsd_reasoning/checkpoint-700-merged"
OUTPUT_DIR="saved_out/mcsd"
NPROC_PER_NODE=4 \
ENABLE_AUDIO_OUTPUT=0 \
USE_AUDIO_IN_VIDEO=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type grpo \
    --model "$CHECKPOINT_PATH" \
    --resume_from_checkpoint "$RESUME_PATH" \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs accuracy_reward format_reward \
    --reward_model "$RM_PATH" \
    --reward_model_plugin my_custom_genrm \
    --reward_weights 1.0 0.5 0.5 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "$DATASET_PATH" \
    --val_dataset "$VAL_DATASET_PATH" \
    --load_from_cache_file true \
    --max_completion_length 512 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2\
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 50 \
    --logging_steps 5 \
    --max_length 12280 \
    --output_dir "$OUTPUT_DIR/grpo_reasoning_genrm1" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1. \
    --top_p 0.99 \
    --top_k 50 \
    --system 'examples/train/grpo/prompt.txt' \
    --log_completions true \
    --report_to wandb