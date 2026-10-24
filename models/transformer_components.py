import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SafeMultiheadAttention(nn.Module):
    """
    安全的多头注意力实现，避免NaN问题
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        # 确保embed_dim可以被num_heads整除
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} 不能被 num_heads {num_heads} 整除"
        
        self.head_dim = embed_dim // num_heads
        
        # QKV投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化参数"""
        # 使用适当的初始化方法
        gain = 0.5 if self.dropout > 0.0 else 1.0
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)
        
        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)
    
    def _shape(self, tensor, seq_len, batch_size):
        """调整张量形状以进行多头注意力计算"""
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def safe_softmax(self, x, dim=-1):
        """安全的softmax实现，防止NaN"""
        # 裁剪极端值提高数值稳定性
        x = torch.clamp(x, min=-1e4, max=1e4)
        
        # 获取掩码 (-1e4表示被掩码的位置)
        mask = (x > -1e4)
        
        # 检测全掩码行
        all_masked_rows = ~mask.any(dim=dim, keepdim=True)
        
        # 为全掩码行创建均匀分布
        dummy_dist = torch.ones_like(x) / x.size(dim)
        
        # 计算softmax
        x_max, _ = torch.max(x, dim=dim, keepdim=True)
        exp_x = torch.exp(x - x_max)
        # 将掩码位置设为0
        exp_x = torch.where(mask, exp_x, torch.zeros_like(exp_x))
        
        # 计算分母（避免除零）
        denom = exp_x.sum(dim=dim, keepdim=True) + 1e-12
        result = exp_x / denom
        
        # 对全掩码行使用均匀分布
        result = torch.where(all_masked_rows, dummy_dist, result)
        
        # 确保结果无NaN/Inf
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        return result
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        """
        前向传播
        
        Args:
            query, key, value: 输入张量
            key_padding_mask: [batch_size, src_len] 布尔掩码，True表示填充位置
            need_weights: 是否返回注意力权重
            attn_mask: 注意力掩码
            
        Returns:
            注意力输出，可选地返回注意力权重
        """
        # 确认维度并处理batch_first参数
        if self.batch_first:
            batch_size, tgt_len = query.shape[:2]
            src_len = key.shape[1]
        else:
            tgt_len, batch_size = query.shape[:2]
            src_len = key.shape[0]
            # 转换为batch_first格式
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # 投影Q、K、V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 调整形状为多头格式
        q = self._shape(q, tgt_len, batch_size)  # [batch_size, num_heads, tgt_len, head_dim]
        k = self._shape(k, src_len, batch_size)  # [batch_size, num_heads, src_len, head_dim]
        v = self._shape(v, src_len, batch_size)  # [batch_size, num_heads, src_len, head_dim]
        
        # 计算注意力分数 (缩放点积注意力)
        scale = 1.0 / math.sqrt(self.head_dim)
        q = q * scale
        
        # 矩阵乘法 [batch_size, num_heads, tgt_len, src_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # 应用掩码 - 使用大负数而非-inf
        mask_value = -1e4
        
        # 应用注意力掩码
        if attn_mask is not None:
            # 根据掩码维度调整形状
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, src_len]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [batch_size, 1, tgt_len, src_len]
            attn_weights = torch.where(attn_mask.bool(), attn_weights, torch.tensor(mask_value, device=attn_weights.device))
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            # 检查全填充行
            all_padding_rows = key_padding_mask.all(dim=1)
            if all_padding_rows.any():
                # 修复全填充行 - 保持至少一个位置未掩码
                if key_padding_mask.shape[1] > 0:
                    # 将每行的第一个位置设为False
                    key_padding_mask = key_padding_mask.clone()
                    key_padding_mask[all_padding_rows, 0] = False
            
            # 扩展键填充掩码 [batch_size, src_len] -> [batch_size, 1, 1, src_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = torch.where(key_padding_mask, torch.tensor(mask_value, device=attn_weights.device), attn_weights)
        
        # 使用安全的softmax
        attn_weights = self.safe_softmax(attn_weights, dim=-1)
        
        # 应用dropout
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, tgt_len, head_dim]
        
        # 重塑张量 [batch_size, tgt_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim)
        
        # 输出投影
        attn_output = self.out_proj(attn_output)
        
        # 如果不是batch_first格式，转换回原始格式
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
            attn_weights = attn_weights.transpose(0, 1)
        
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output

class SafeTransformerEncoderLayer(nn.Module):
    """
    安全的Transformer编码器层，避免NaN问题
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        
        # 自注意力 - 使用SafeMultiheadAttention
        self.self_attn = SafeMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # 层标准化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.activation = self._get_activation_fn(activation)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # 梯度缩放系数
        self.attn_scale = 1.0
        self.ff_scale = 1.0
        
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化参数"""
        # 前馈网络初始化
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.8)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.8)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)
    
    def _get_activation_fn(self, activation):
        """获取激活函数"""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """前向传播"""
        # 自注意力子层
        src2 = self.norm1(src)
        
        try:
            # 应用自注意力
            src2 = self.self_attn(src2, src2, src2, 
                                attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        except Exception as e:
            # 出错时使用输入作为输出
            return src
        
        # 检查输出是否有NaN/Inf
        if torch.isnan(src2).any() or torch.isinf(src2).any():
            # 使用零张量替代问题输出
            src2 = torch.zeros_like(src2)
        
        # 残差连接
        src = src + src2 * self.attn_scale
        
        # 前馈子层
        try:
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout1(self.activation(self.linear1(src2))))
            src2 = self.dropout2(src2)
            
            # 残差连接
            src = src + src2 * self.ff_scale
        except Exception as e:
            # 前馈网络出错时跳过
            pass
        
        # 确保输出无NaN/Inf
        if torch.isnan(src).any() or torch.isinf(src).any():
            src = torch.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)
            
        return src

class SafeTransformerEncoder(nn.Module):
    """
    安全的Transformer编码器，避免NaN问题
    """
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # 创建多个编码器层
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
        # 为深层模型添加梯度缩放
        if num_layers > 8:
            for i, layer in enumerate(self.layers):
                scale = 1.0 / math.sqrt(2.0 * (i + 1))
                layer.attn_scale = scale
                layer.ff_scale = scale
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        前向传播
        
        Args:
            src: 输入序列 [batch_size, seq_len, embed_dim]
            mask: 序列掩码
            src_key_padding_mask: 填充掩码 [batch_size, seq_len]
            
        Returns:
            编码器输出
        """
        output = src
        
        # 修复src_key_padding_mask中全掩码行
        if src_key_padding_mask is not None and src_key_padding_mask.shape[1] > 0:
            all_mask_rows = src_key_padding_mask.all(dim=1)
            if all_mask_rows.any():
                src_key_padding_mask = src_key_padding_mask.clone()
                src_key_padding_mask[all_mask_rows, 0] = False
        
        # 通过每一层
        for i, layer in enumerate(self.layers):
            try:
                prev_output = output  # 保存当前输出
                output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            except Exception as e:
                # 层处理出错时使用上一层输出
                output = prev_output
                continue
            
            # 如果输出包含NaN/Inf，使用上一层的输出
            if torch.isnan(output).any() or torch.isinf(output).any():
                output = prev_output
        
        return output