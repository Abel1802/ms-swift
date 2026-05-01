#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:8
#SBATCH --time=10:20:00
#SBATCH --output=run_logs/mcsd_reasoning_grpo_all_genrm1.log


module load Miniconda3
source activate ms
module load CUDA/12.8.0

bash examples/train/grpo/qwen2_5_omni/grpo_mcsd.sh