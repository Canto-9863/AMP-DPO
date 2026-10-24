import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split

def preprocess_for_training(input_csv, output_dir, train_ratio=0.98, 
                           val_ratio=0.01, test_ratio=0.01, batch_size=32, 
                           max_length=100, seed=42):
    """将预处理数据转换为训练数据集格式"""
    
    print(f"读取数据: {input_csv}")
    df = pd.read_csv(input_csv)
    
    df = df[df['sequence'].str.len() <= max_length]
    print(f"过滤后剩余序列: {len(df)}")
    
    train, temp = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=seed)
    ratio = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(temp, test_size=(1-ratio), random_state=seed)
    
    print(f"训练集大小: {len(train)}")
    print(f"验证集大小: {len(val)}")
    print(f"测试集大小: {len(test)}")
    
    def batch_by_length(dataset, batch_size):
        dataset = dataset.copy()
        dataset['length'] = dataset['sequence'].str.len()
        dataset = dataset.sort_values('length')
        
        batches = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset.iloc[i:i+batch_size]
            batches.append(batch[['sequence']])
        
        return batches
    
    train_batches = batch_by_length(train, batch_size)
    val_batches = batch_by_length(val, batch_size)
    test_batches = batch_by_length(test, batch_size)
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print(f"保存训练批次...")
    for i, batch in enumerate(tqdm(train_batches)):
        batch.to_csv(os.path.join(train_dir, f'batch_{i:05d}.csv'), index=False)
    
    print(f"保存验证批次...")
    for i, batch in enumerate(tqdm(val_batches)):
        batch.to_csv(os.path.join(val_dir, f'batch_{i:05d}.csv'), index=False)
    
    print(f"保存测试批次...")
    for i, batch in enumerate(tqdm(test_batches)):
        batch.to_csv(os.path.join(test_dir, f'batch_{i:05d}.csv'), index=False)
    
    train_index = os.path.join(train_dir, 'index.csv')
    val_index = os.path.join(val_dir, 'index.csv')
    test_index = os.path.join(test_dir, 'index.csv')
    
    pd.DataFrame({'file': [f'batch_{i:05d}.csv' for i in range(len(train_batches))]}).to_csv(train_index, index=False)
    pd.DataFrame({'file': [f'batch_{i:05d}.csv' for i in range(len(val_batches))]}).to_csv(val_index, index=False)
    pd.DataFrame({'file': [f'batch_{i:05d}.csv' for i in range(len(test_batches))]}).to_csv(test_index, index=False)
    
    print(f"数据准备完成:")
    print(f"  训练批次数: {len(train_batches)}")
    print(f"  验证批次数: {len(val_batches)}")
    print(f"  测试批次数: {len(test_batches)}")
    print(f"  总批次数: {len(train_batches) + len(val_batches) + len(test_batches)}")
    
    return {
        'train_batches': len(train_batches),
        'val_batches': len(val_batches),
        'test_batches': len(test_batches),
        'train_samples': len(train),
        'val_samples': len(val),
        'test_samples': len(test)
    }

def main():
    parser = argparse.ArgumentParser(description='为训练准备UniProt数据')
    parser.add_argument('--input', type=str, required=True,
                       help='输入CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='data/training_data',
                       help='输出目录')
    parser.add_argument('--max_length', type=int, default=100,
                       help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--train_ratio', type=float, default=0.98,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.01,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.01,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    args = parser.parse_args()
    
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
        print(f"警告: 比例之和不为1, 已归一化为 train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    
    preprocess_for_training(
        args.input,
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed
    )

if __name__ == '__main__':
    main()