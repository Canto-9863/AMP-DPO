
python generate_discrete.py ^
    --checkpoint checkpoints/discrete_dpo/discrete_amp_dpo_best.pt ^
    --num_samples 300 ^
    --num_strategies 10 ^
    --iterations 3 ^
    --batch_size 10 ^
    --output_dir 8generated_results ^
    --output_prefix discrete_amps ^
    --top_k 100 ^
    --seq_len_min 2 ^
    --seq_len_max 11