#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH -t 14:00:00
#SBATCH -J r2
# MODEL = local directory of Qwen3Guard-Gen-4B (download it first; compute nodes have no network)
# run from this folder after ./setup.sh
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python 12_sft.py --model "$MODEL" --tag r2 --train_csv train_r2.csv --many_shot_aug 0.3
