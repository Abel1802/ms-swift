CHECKPOINT_PATH="saved_out/my_genrm_3b_mcsd_reasoning/checkpoint-700"
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters "$CHECKPOINT_PATH" \
    --merge_lora true