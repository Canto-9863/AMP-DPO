import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class AMPPreferenceDataset(Dataset):
    """抗菌肽偏好对数据集"""
    
    def __init__(self, data_path, tokenizer, max_length=50):
        """
        初始化数据集
        
        Args:
            data_path: CSV文件路径，包含偏好对数据
            tokenizer: 用于序列标记化的分词器
            max_length: 序列的最大长度
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 确保偏好对的正确性（win序列应该优于lose序列）
        valid_pairs = self.data['mic_preferred'] < self.data['mic_non_preferred']
        self.data = self.data[valid_pairs].reset_index(drop=True)
        
        print(f"加载了{len(self.data)}个有效的偏好对")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取一个偏好对样本"""
        row = self.data.iloc[idx]
        
        # 编码序列
        win_seq = self.tokenizer.encode(
            row['seq_preferred'], 
            max_length=self.max_length
        )
        
        lose_seq = self.tokenizer.encode(
            row['seq_non_preferred'], 
            max_length=self.max_length
        )
        
        # 提取属性特征（如果有）
        properties = []
        for prop in ['charge', 'hydrophobicity', 'length']:
            if f'{prop}_preferred' in row:
                properties.append(row[f'{prop}_preferred'])
        
        sample = {
            'win_seq': win_seq,
            'lose_seq': lose_seq,
        }
        
        if properties:
            sample['properties'] = torch.tensor(properties, dtype=torch.float32)
            
        return sample


class SyntheticAMPDataset(Dataset):
    """合成抗菌肽数据集（用于测试）"""
    
    def __init__(self, tokenizer, num_samples=1000, max_length=50, seed=42):
        """
        创建合成数据集
        
        Args:
            tokenizer: 用于序列标记化的分词器
            num_samples: 样本数量
            max_length: 最大序列长度
            seed: 随机种子
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        np.random.seed(seed)
        
        # 创建合成数据
        self.data = []
        
        # 可用氨基酸列表
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        for i in range(num_samples):
            # 生成随机序列长度
            win_len = np.random.randint(10, 30)
            lose_len = np.random.randint(10, 30)
            
            # 生成随机序列
            win_seq = ''.join(np.random.choice(list(amino_acids), size=win_len))
            lose_seq = ''.join(np.random.choice(list(amino_acids), size=lose_len))
            
            # 生成随机属性
            charge_win = np.random.uniform(-5, 5)
            charge_lose = np.random.uniform(-5, 5)
            
            hydrophobicity_win = np.random.uniform(0, 1)
            hydrophobicity_lose = np.random.uniform(0, 1)
            
            # 确保win_seq具有更好的属性（更高的抗菌活性，更低的MIC值）
            mic_win = np.random.uniform(0.1, 10)
            mic_lose = mic_win * np.random.uniform(1.2, 3.0)  # 确保mic_lose > mic_win
            
            self.data.append({
                'seq_preferred': win_seq,
                'seq_non_preferred': lose_seq,
                'mic_preferred': mic_win,
                'mic_non_preferred': mic_lose,
                'charge_preferred': charge_win,
                'charge_non_preferred': charge_lose,
                'hydrophobicity_preferred': hydrophobicity_win,
                'hydrophobicity_non_preferred': hydrophobicity_lose,
                'length_preferred': win_len,
                'length_non_preferred': lose_len
            })
        
        self.data = pd.DataFrame(self.data)
        print(f"创建了{len(self.data)}个合成偏好对")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取一个合成偏好对样本"""
        row = self.data.iloc[idx]
        
        # 编码序列
        win_seq = self.tokenizer.encode(
            row['seq_preferred'], 
            max_length=self.max_length
        )
        
        lose_seq = self.tokenizer.encode(
            row['seq_non_preferred'], 
            max_length=self.max_length
        )
        
        # 提取属性
        properties = torch.tensor([
            row['charge_preferred'],
            row['hydrophobicity_preferred'],
            row['length_preferred']
        ], dtype=torch.float32)
        
        return {
            'win_seq': win_seq,
            'lose_seq': lose_seq,
            'properties': properties
        }
    
    def save_to_csv(self, file_path):
        """将合成数据保存到CSV文件"""
        self.data.to_csv(file_path, index=False)
        print(f"合成数据已保存到 {file_path}")