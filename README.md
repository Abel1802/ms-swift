# Generate reasoning for a given multimodal dataset for sarcasm detection and reasoning

# Usage

## Step1. Prepare json file for infer & sample(videos can't be too long)
```bash
python examples/train/grpo/qwen2_5_omni/process_data_mcsd.py
```

## Step2. Run inference & sampling
```bash
bash examples/train/grpo/qwen3omni/infer_vllm.sh
```

```bash
bash examples/train/grpo/qwen3omni/sample.sh
```


## Step3. Check the F1-score & ACC
```bash for inference
python examples/train/grpo/qwen3omni/cal_f1_omni3.py
```

```bash for sampling
python cal_acc_sample.py
```

## Ste4. Filter Best-of-N & all_correct of Sampling & Greedy-deceding of Inference
```bash for top1 sampling
python examples/train/grpo/qwen3omni/baseline_filter.py
```

```bash for all_correct sampling
python examples/train/grpo/qwen3omni/all_filter.py
```

```bash for greedy deceding
python examples/train/grpo/qwen3omni/construct_data2.py
```

## Step5. SFT
```bash
bash examples/train/grpo/qwen2_5_omni/sft_mcsd.sh
```

## Step6. Merge checkpoints
```bash
bash examples/train/grpo/qwen2_5_omni/merge.sh
```

## Step7. GRPO
```bash
bash examples/train/grpo/qwen2_5_omni/grpo_mcsd_vllm.sh
```


## Step8. SFT GenRM
```bash
python examples/train/grpo/qwen3omni/rm_filter.py
bash examples/train/grpo/qwen2_5_omni/sft_genrm.sh
```

## Step9. GRPO w/ GenRM
```bash
bash examples/train/grpo/qwen2_5_omni/grpo_mstdpp_vllm_rm.sh
```
