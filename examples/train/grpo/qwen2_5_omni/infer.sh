CHECKPOINT_PATH="shared_output/omni_sft_a_v_omni3_think_baseline_reason_all/v0-20260220-152855/checkpoint-404-merged"
VAL_DATASET_PATH="raw_data/MCSD1/ms_swift_video_audio/test_zh.jsonl"
SAVE_DIR="examples/train/grpo/qwen2_5_omni"
RESULTS_PATH="$SAVE_DIR/test_pred.jsonl"
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen2.5-Omni-7B \
    --val_dataset "$VAL_DATASET_PATH" \
    --result_path "$RESULTS_PATH" \
    --temperature 0.0 \
    --top_p 1.0 \
    --infer_backend vllm \
    --vllm_max_model_len 12280