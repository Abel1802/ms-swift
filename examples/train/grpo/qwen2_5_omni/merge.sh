CHECKPOINT_PATH="saved_out/mcsd/grpo_reasoning_genrm1/v7-20260502-100411/checkpoint-700"
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters "$CHECKPOINT_PATH" \
    --merge_lora true