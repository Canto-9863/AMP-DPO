@echo off
set CUDA_VISIBLE_DEVICES=0

python train_discrete.py ^
    --data_path data/amp_preference_train.csv ^
    --val_data_path data/amp_preference_val.csv ^
    --pretrained_model checkpoints/discrete_pretrain/discrete_diffusion_pretrained_latest.pt ^
    --num_epochs 60 ^
    --batch_size 128 ^
    --learning_rate 5e-3 ^
    --patience 20 ^
    --progressive_beta ^
    --initial_beta 0.1 ^
    --beta 1.0