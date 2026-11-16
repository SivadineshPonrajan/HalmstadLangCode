#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -p gpu
#SBATCH --gpus=4
#SBATCH -c 64
#SBATCH -t 08:00:00
#SBATCH -J fft
# Run from the repo root. Needs ../lora/setup.sh to have been run, and DeepSpeed in ./ds_pkgs (see fft/README.md).
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTHONPATH=$PWD/ds_pkgs:$PWD/lora
cd lora
python -m torch.distributed.run --standalone --nproc_per_node=4 ../fft/41_fft_ds.py \
    --model "$MODEL" --tag fft --train_csv train_r2.csv --many_shot_aug 0.3
