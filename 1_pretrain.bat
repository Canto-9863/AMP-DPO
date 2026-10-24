@echo off
set CUDA_VISIBLE_DEVICES=0

python pretrain_discrete.py ^
    --train_data pretrain_data/data/training_data/train ^
    --val_data pretrain_data/data/training_data/val ^
    --embedding_dim 128 ^
    --hidden_dim 512 ^
    --transformer_layers 6 ^
    --transformer_heads 8 ^
    --num_timesteps 1000 ^
    --beta_schedule linear ^
    --batch_size 512 ^
    --num_epochs 60 ^
    --learning_rate 1e-4 ^
    --checkpoint_dir checkpoints/discrete_pretrain ^