import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from Bio import SeqIO
import re
import pickle

def clean_sequence(seq):
    """清理序列中的非标准氨基酸字符"""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    return ''.join(c for c in seq if c in valid_aa)

def process_fasta(fasta_path, output_dir, max_length=1000, min_length=20, chunk_size=10000):
    """处理FASTA文件，分块保存为CSV"""
    
    print(f"处理FASTA文件: {fasta_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    total_sequences = 0
    valid_sequences = 0
    length_distribution = {}
    
    sequences = []
    chunk_idx = 0
    
    with open(fasta_path, 'r') as handle:
        total_count = 0
        for line in handle:
            if line.startswith('>'):
                total_count += 1
        print(f"文件包含 {total_count} 个序列")
    
    with open(fasta_path, 'r') as handle:
        seq_iterator = SeqIO.parse(handle, 'fasta')
        
        for record in tqdm(seq_iterator, total=total_count, desc="处理序列"):
            total_sequences += 1
            
            seq_str = str(record.seq)
            
            clean_seq = clean_sequence(seq_str)
            seq_len = len(clean_seq)
            
            if min_length <= seq_len <= max_length:
                sequences.append({
                    'id': record.id,
                    'sequence': clean_seq,
                    'length': seq_len
                })
                valid_sequences += 1
                
                length_bin = seq_len // 50 * 50  
                length_distribution[length_bin] = length_distribution.get(length_bin, 0) + 1
            
            if len(sequences) >= chunk_size:
                chunk_path = os.path.join(output_dir, f'protein_chunk_{chunk_idx}.csv')
                pd.DataFrame(sequences).to_csv(chunk_path, index=False)
                print(f"保存分块 {chunk_idx}: {len(sequences)} 条序列")
                sequences = []
                chunk_idx += 1
    
    if sequences:
        chunk_path = os.path.join(output_dir, f'protein_chunk_{chunk_idx}.csv')
        pd.DataFrame(sequences).to_csv(chunk_path, index=False)
        print(f"保存分块 {chunk_idx}: {len(sequences)} 条序列")
    
    stats = {
        'total_sequences': total_sequences,
        'valid_sequences': valid_sequences,
        'length_distribution': dict(sorted(length_distribution.items()))
    }
    
    stats_path = os.path.join(output_dir, 'stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    length_df = pd.DataFrame([
        {'length_bin': k, 'count': v} 
        for k, v in sorted(length_distribution.items())
    ])
    length_df.to_csv(os.path.join(output_dir, 'length_distribution.csv'), index=False)
    
    print(f"处理完成: 共 {total_sequences} 条序列，有效序列 {valid_sequences} 条")
    print(f"统计信息已保存到 {stats_path}")
    
    return output_dir

def merge_chunks(chunks_dir, output_path, sample_ratio=1.0, max_samples=None, seed=42):
    """合并或抽样多个数据块为单个训练集"""
    all_chunks = [f for f in os.listdir(chunks_dir) if f.startswith('protein_chunk_') and f.endswith('.csv')]
    all_chunks.sort()
    
    np.random.seed(seed)
    
    total_sequences = 0
    sampled_sequences = []
    
    print(f"合并 {len(all_chunks)} 个数据块...")
    
    for chunk_file in tqdm(all_chunks, desc="计算总数"):
        chunk_path = os.path.join(chunks_dir, chunk_file)
        chunk_df = pd.read_csv(chunk_path)
        total_sequences += len(chunk_df)
    
    print(f"总序列数: {total_sequences}")
    
    if sample_ratio < 1.0:
        sample_size = int(total_sequences * sample_ratio)
        print(f"将抽样 {sample_ratio:.1%} = {sample_size} 条序列")
    elif max_samples and max_samples < total_sequences:
        sample_ratio = max_samples / total_sequences
        sample_size = max_samples
        print(f"将抽样最多 {sample_size} 条序列 ({sample_ratio:.1%})")
    else:
        sample_size = total_sequences
        print(f"将使用全部 {sample_size} 条序列")
    
    current_count = 0
    for chunk_file in tqdm(all_chunks, desc="抽样数据"):
        chunk_path = os.path.join(chunks_dir, chunk_file)
        chunk_df = pd.read_csv(chunk_path)
        
        chunk_df = chunk_df[['sequence']]
        
        chunk_size = len(chunk_df)
        chunk_ratio = chunk_size / total_sequences
        chunk_sample_size = int(chunk_ratio * sample_size)
        
        if chunk_sample_size < chunk_size:
            chunk_df = chunk_df.sample(chunk_sample_size, random_state=seed+current_count)
        
        sampled_sequences.append(chunk_df)
        current_count += 1
    
    final_df = pd.concat(sampled_sequences, ignore_index=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"抽样和合并完成: {len(final_df)} 条序列已保存到 {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='处理UniProt FASTA数据')
    parser.add_argument('--input', type=str, required=True,
                       help='输入FASTA文件路径')
    parser.add_argument('--output_dir', type=str, default='data/processed_uniprot',
                       help='输出目录')
    parser.add_argument('--min_length', type=int, default=20,
                       help='最小序列长度')
    parser.add_argument('--max_length', type=int, default=500,
                       help='最大序列长度')
    parser.add_argument('--chunk_size', type=int, default=50000,
                       help='每个分块的大小')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='抽样比例 (0-1)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大抽样数量')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    args = parser.parse_args()
    
    chunks_dir = os.path.join(args.output_dir, 'chunks')
    process_fasta(
        args.input, 
        chunks_dir, 
        max_length=args.max_length, 
        min_length=args.min_length,
        chunk_size=args.chunk_size
    )
    
    final_output = os.path.join(args.output_dir, 'protein_sequences.csv')
    merge_chunks(
        chunks_dir, 
        final_output,
        sample_ratio=args.sample_ratio,
        max_samples=args.max_samples,
        seed=args.seed
    )

if __name__ == '__main__':
    main()