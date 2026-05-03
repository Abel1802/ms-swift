#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:4
#SBATCH --time=23:50:00
#SBATCH --output=run_logs/mcsd_reasoning_grpo_all_genrm1.log


module load Miniconda3
module load CUDA/12.8.0
source activate ms

bash examples/train/grpo/qwen2_5_omni/grpo_mcsd.sh