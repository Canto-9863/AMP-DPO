import numpy as np
import torch

class ProteinTokenizer:
    """氨基酸序列分词器"""
    
    def __init__(self):
        """初始化分词器"""
        # 标准20种氨基酸的单字母编码
        self.aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        
        # 特殊标记
        self.pad_token = "<PAD>"  # 填充标记
        self.unk_token = "<UNK>"  # 未知氨基酸标记
        self.start_token = "<START>"  # 起始标记
        self.end_token = "<END>"  # 结束标记
        
        # 构建氨基酸到索引的映射
        special_tokens = [self.pad_token, self.unk_token, self.start_token, self.end_token]
        self.token_to_id = {token: i for i, token in enumerate(special_tokens + self.aa_list)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        self.pad_id = self.token_to_id[self.pad_token]
        self.unk_id = self.token_to_id[self.unk_token]
        self.start_id = self.token_to_id[self.start_token]
        self.end_id = self.token_to_id[self.end_token]
        
        self.vocab_size = len(self.token_to_id)
    
    def encode(self, sequence, add_special_tokens=True, max_length=None):
        """
        将氨基酸序列转换为ID序列
        
        Args:
            sequence: 氨基酸序列字符串
            add_special_tokens: 是否添加起始和结束标记
            max_length: 最大序列长度（包括特殊标记）
            
        Returns:
            tensor: ID序列的张量
        """
        if not sequence:
            return torch.tensor([self.pad_id])
        
        # 转换氨基酸为ID
        ids = []
        
        if add_special_tokens:
            ids.append(self.start_id)
        
        for aa in sequence:
            if aa in self.token_to_id:
                ids.append(self.token_to_id[aa])
            else:
                ids.append(self.unk_id)
        
        if add_special_tokens:
            ids.append(self.end_id)
        
        # 处理最大长度
        if max_length is not None:
            if len(ids) > max_length:
                # 截断序列，但保留结束标记
                if add_special_tokens:
                    ids = ids[:max_length-1] + [self.end_id]
                else:
                    ids = ids[:max_length]
            else:
                # 填充序列到最大长度
                padding_length = max_length - len(ids)
                ids = ids + [self.pad_id] * padding_length
        
        return torch.tensor(ids)
    
    def decode(self, ids, skip_special_tokens=True):
        """
        将ID序列转换回氨基酸序列
        
        Args:
            ids: ID序列（列表或张量）
            skip_special_tokens: 是否跳过特殊标记
            
        Returns:
            str: 解码后的氨基酸序列
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()
        
        sequence = []
        for idx in ids:
            token = self.id_to_token.get(idx, self.unk_token)
            if skip_special_tokens and token in [self.pad_token, self.start_token, self.end_token]:
                continue
            sequence.append(token)
        
        return "".join(sequence)
    
    def batch_decode(self, batch_ids, skip_special_tokens=True):
        """批量解码多个序列"""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
    
    def get_embedding_weights(self, embedding_dim=512):
        """
        为嵌入层创建初始化权重
        这里使用简单的随机初始化，实际应用中可以使用基于BLOSUM矩阵或其他氨基酸相似性的初始化
        """
        weights = torch.randn(self.vocab_size, embedding_dim)
        return weights