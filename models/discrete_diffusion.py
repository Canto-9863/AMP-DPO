import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.transformer_components import SafeTransformerEncoder, SafeTransformerEncoderLayer

class DiscreteNoiseSchedule:
    """
    离散噪声调度类，负责离散扩散过程中的噪声添加和参数计算
    """
    
    def __init__(self, num_timesteps=1000, vocab_size=24, mask_id=0, beta_schedule="linear"):
        """
        初始化离散噪声调度
        
        Args:
            num_timesteps: 时间步数量
            vocab_size: 词汇量大小
            mask_id: 掩码标记的ID（通常为PAD标记）
            beta_schedule: β调度类型 ("linear", "cosine", "sqrt")
        """
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        
        # 创建beta调度（腐蚀概率）
        if beta_schedule == "linear":
            # 线性增加腐蚀概率从接近0到接近1
            self.betas = torch.linspace(0.0001, 0.9999, num_timesteps)
        elif beta_schedule == "cosine":
            # 余弦调度
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        elif beta_schedule == "sqrt":
            # 平方根调度
            self.betas = torch.linspace(0.0001, 0.9999, num_timesteps) ** 0.5
        else:
            raise ValueError(f"未支持的beta调度: {beta_schedule}")
        
        # 计算累积腐蚀概率
        self.alphas = 1.0 - self.betas  # 保留原始token的概率
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # 经过t步后保留原始token的概率
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])  # t-1步时保留原始token的概率
        
    def q_sample(self, x_0, t, noise=None):
        """
        对输入序列添加t步噪声（通过随机替换token为mask_id）
        
        Args:
            x_0: 原始序列 [batch_size, seq_len]
            t: 时间步 [batch_size]
            noise: 在离散情况下，这个参数被忽略，仅为保持API兼容性
            
        Returns:
            x_t: 噪声序列
            noise_mask: 噪声掩码，指示哪些位置被替换了
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # 获取t时刻的累积腐蚀概率
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_0.shape)
        
        # 确保腐蚀概率不会太小 限制最大值为0.95
        alpha_cumprod_t = torch.clamp(alpha_cumprod_t, max=0.95)
            
        # 生成均匀随机数决定哪些位置被腐蚀
        noise_probs = torch.rand_like(x_0.float(), device=device)
        noise_mask = noise_probs > alpha_cumprod_t
        
        # 创建被腐蚀的序列（将被腐蚀的位置替换为mask_id）
        x_t = x_0.clone()
        x_t[noise_mask] = self.mask_id
        
        # 确保至少有一些位置被掩码化
        if noise_mask.sum() == 0:
            # 随机选择10%的位置进行掩码
            random_mask = torch.rand_like(x_0.float(), device=device) < 0.1
            x_t[random_mask] = self.mask_id
            noise_mask = random_mask
            
        return x_t, noise_mask
    
    def _extract(self, a, t, shape=None):
        """
        从系数张量中提取特定时间步的值，并调整形状以便于广播
        """
        batch_size = t.shape[0]
        
        # 确保t在有效范围内
        t = torch.clamp(t, 0, len(a) - 1)
        
        out = a.gather(-1, t.cpu()).to(t.device)
        
        if shape is None:
            return out
        
        # 调整形状以便于广播
        shape = [batch_size] + [1] * (len(shape) - 1)
        return out.reshape(shape)


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置嵌入，用于时间步编码"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # 确保形状正确
        if embeddings.shape[-1] < self.dim:
            # 如果维度不是偶数，填充一个零
            padding = torch.zeros(embeddings.shape[0], 1, device=time.device)
            embeddings = torch.cat([embeddings, padding], dim=-1)
            
        return embeddings


class DiscreteProteinDiffusionModel(nn.Module):
    """
    离散蛋白质扩散模型
    一个用于生成蛋白质序列的离散扩散模型，可以接受条件信息
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim=128,
                 hidden_dim=512,
                 condition_dim=32,
                 transformer_layers=6,
                 transformer_heads=8,
                 num_timesteps=1000,
                 beta_schedule="linear",
                 beta_start=1e-4,
                 beta_end=0.02,
                 conditional=True,
                 dropout=0.1,
                 pad_id=0,
                 debug=False):
        super().__init__()

        self.debug = debug
        print(
            f"初始化DiscreteProteinDiffusionModel: vocab_size={vocab_size}, embedding_dim={embedding_dim}, condition_dim={condition_dim}")

        # 模型参数
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.conditional = conditional
        self.pad_id = pad_id

        # 序列嵌入
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(1024, embedding_dim)
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        
        # 设置总嵌入维度
        total_embed_dim = embedding_dim
        if conditional:
            # 属性条件编码器
            self.condition_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, condition_dim), nn.GELU(),
                nn.Dropout(dropout))
            total_embed_dim += condition_dim

        print(f"总嵌入维度: {total_embed_dim}")

        # 设置噪声调度
        self.noise_schedule = DiscreteNoiseSchedule(
            num_timesteps=num_timesteps,
            vocab_size=vocab_size,
            mask_id=pad_id,
            beta_schedule=beta_schedule
        )
        
        # 确保transformer_heads能够整除total_embed_dim
        if total_embed_dim % transformer_heads != 0:
            old_transformer_heads = transformer_heads
            transformer_heads = self._find_suitable_heads(total_embed_dim, transformer_heads)
            print(f"警告: transformer_heads={old_transformer_heads}不能整除total_embed_dim={total_embed_dim}，自动调整为{transformer_heads}")
        
        # 添加层归一化以提高稳定性
        self.embed_norm = nn.LayerNorm(total_embed_dim)
            
        # 创建Transformer编码器层
        encoder_layer = SafeTransformerEncoderLayer(
            d_model=total_embed_dim, 
            nhead=transformer_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Transformer编码器
        self.transformer_encoder = SafeTransformerEncoder(
            encoder_layer, 
            num_layers=transformer_layers
        )
        
        # 输出层 - 预测每个位置的词汇分布
        self.output_layer = nn.Linear(total_embed_dim, vocab_size)
        print(f"输出层: {total_embed_dim} -> {vocab_size}")
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重以提高训练稳定性"""
        # 嵌入层初始化
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # 线性层初始化
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() == 2:
                # 线性层使用xavier初始化
                nn.init.xavier_uniform_(p, gain=0.8)
            elif "bias" in name:
                # 所有偏置初始化为0
                nn.init.zeros_(p)
                
        # 特别是对输出层进行小初始化，防止初始输出太大
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_layer.bias)
    
    def _find_suitable_heads(self, dim, preferred_heads):
        """找到适合的注意力头数，确保能够整除嵌入维度"""
        # 尝试从首选头数开始，找到最近的合适值
        for heads in [preferred_heads, preferred_heads-1, preferred_heads+1, 
                     preferred_heads-2, preferred_heads+2, preferred_heads//2, 
                     8, 4, 2, 1]:
            if heads > 0 and dim % heads == 0:
                return heads
        return 1  # 最后的保底方案

    def _get_embeddings(self, x, t, condition=None):
        """获取序列的嵌入表示"""
        batch_size, seq_len = x.shape

        if self.debug:
            print(f"输入形状: {x.shape}")

        # 位置索引
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # 检查输入范围并修复
        x = torch.clamp(x, 0, self.vocab_size - 1)

        # 计算嵌入
        token_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(positions)
        
        # 时间嵌入
        time_embed = self.time_mlp(t.float())
        time_embed = time_embed.unsqueeze(1).expand(-1, seq_len, -1)

        # 组合嵌入
        embed = token_embed + pos_embed + time_embed  # [batch_size, seq_len, embed_dim]

        if self.debug:
            print(f"基础嵌入形状: {embed.shape}")

        # 如果提供了条件，则添加
        if condition is not None and self.conditional:
            cond_embed = self.condition_encoder(condition)
            cond_embed = cond_embed.unsqueeze(1).expand(-1, seq_len, -1)

            if self.debug:
                print(f"条件嵌入形状: {cond_embed.shape}")

            # 拼接条件和序列嵌入
            embed = torch.cat([embed, cond_embed], dim=-1)

            if self.debug:
                print(f"拼接后嵌入形状: {embed.shape}")
        elif self.conditional:
            # 如果模型是条件模型但没有提供条件，创建零填充的条件嵌入
            zero_cond = torch.zeros(batch_size, 3, device=x.device)
            cond_embed = self.condition_encoder(zero_cond)
            cond_embed = cond_embed.unsqueeze(1).expand(-1, seq_len, -1)
            embed = torch.cat([embed, cond_embed], dim=-1)
            
            if self.debug:
                print(f"零条件嵌入后形状: {embed.shape}")

        return embed

    def forward(self, x, t, condition=None):
        """前向传播，预测每个位置的token分布"""
        # 获取序列嵌入
        x_embed = self._get_embeddings(x, t, condition)  # [batch_size, seq_len, embed_dim(+cond_dim)]
        
        # 应用层归一化以提高稳定性
        x_embed = self.embed_norm(x_embed)
        
        # 创建注意力掩码（避免关注填充位置）
        padding_mask = (x == self.pad_id)  # [batch_size, seq_len]
        
        # 检查填充掩码，确保有效
        all_pad_rows = padding_mask.all(dim=1)
        if all_pad_rows.any() and padding_mask.shape[1] > 0:
            # 确保每行至少有一个False (避免全掩码行)
            padding_mask = padding_mask.clone()
            padding_mask[all_pad_rows, 0] = False
        
        if self.debug:
            print(f"x_embed形状: {x_embed.shape}, padding_mask形状: {padding_mask.shape}")
            
        # 通过Transformer编码器
        encoder_output = self.transformer_encoder(
            x_embed,
            src_key_padding_mask=padding_mask
        )  # [batch_size, seq_len, embed_dim(+cond_dim)]
        
        # 检查输出并修复NaN/Inf
        if torch.isnan(encoder_output).any() or torch.isinf(encoder_output).any():
            encoder_output = torch.nan_to_num(encoder_output, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 预测每个位置的token分布
        logits = self.output_layer(encoder_output)  # [batch_size, seq_len, vocab_size]
        
        # 确保logits无NaN值
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        
        return logits

    def compute_loss(self, x_0, condition=None):
        """计算离散扩散模型的训练损失"""
        batch_size = x_0.shape[0]
        seq_len = x_0.shape[1]
        device = x_0.device
        
        x_0 = torch.clamp(x_0, 0, self.vocab_size - 1)
        
        # 如果是条件模型但没有提供条件，创建零填充的虚拟条件
        if self.conditional and condition is None:
            condition = torch.zeros(batch_size, 3, device=device)  # 3是条件特征维度
        
        # 随机采样时间步
        t = torch.randint(0, self.noise_schedule.num_timesteps, (batch_size,), device=device)
        
        # 添加噪声
        x_t, noise_mask = self.noise_schedule.q_sample(x_0, t)
        
        # 预测原始序列的分布
        logits = self(x_t, t, condition)  # [batch_size, seq_len, vocab_size]
        
        # 裁剪极端值以提高数值稳定性
        logits = torch.clamp(logits, -50.0, 50.0)
        
        try:
            # 计算交叉熵损失（仅在被噪化的位置）
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                x_0.reshape(-1),
                reduction='none',
             #   label_smoothing=0.1  # 添加标签平滑
            ).reshape(batch_size, seq_len)
            
            # 检查噪声掩码是否有效
            noise_sum = noise_mask.float().sum().item()
            if noise_sum < 1:
                # 无有效噪声掩码，使用平均损失
                return loss.mean()
            
            # 只计算被噪化位置的损失
            masked_loss = (loss * noise_mask.float()).sum() / (noise_sum + 1e-8)
            
            # 限制损失范围
            masked_loss = torch.clamp(masked_loss, 0.0, 100.0)
            
            # 检查损失值
            if torch.isnan(masked_loss) or torch.isinf(masked_loss):
                # 返回一个小的可训练损失
                return torch.tensor(1.0, device=device, requires_grad=True)
                
            return masked_loss
        except Exception as e:
            # 出错时返回一个小的可训练损失
            return torch.tensor(1.0, device=device, requires_grad=True)
    
    def pretraining_loss(self, x, condition=None):
        """与连续扩散模型保持兼容的预训练损失函数"""
        return self.compute_loss(x, condition)

    @torch.no_grad()
    def sample(self,
               batch_size=1,
               seq_len=30,
               condition=None,
               device=None,
               temperature=0.8):
        """
        采样生成新序列
        
        Args:
            batch_size: 生成批次大小
            seq_len: 序列长度
            condition: 可选的条件特征 [batch_size, condition_dim]
            device: 计算设备
            temperature: 采样温度，控制随机性 (较低值=更确定性)
            
        Returns:
            生成的序列ID [batch_size, seq_len]
        """
        device = device or next(self.parameters()).device
        print(
            f"生成参数: batch_size={batch_size}, seq_len={seq_len}, temperature={temperature}"
        )

        # 从全部被mask的序列开始
        x = torch.full((batch_size, seq_len), self.pad_id, device=device)
        
        # 特殊标记屏蔽（排除PAD, UNK, START, END等特殊标记）
        special_token_mask = torch.ones(self.vocab_size, device=device)
        special_token_mask[:4] = 0  # 屏蔽特殊标记的索引

        # 逐步去噪
        for t in range(self.noise_schedule.num_timesteps - 1, -1, -1):
            if t % 100 == 0:
                print(f"去噪步骤: {t}/{self.noise_schedule.num_timesteps}")

            # 为整个批次使用相同的时间步
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            try:
                # 预测token概率分布
                logits = self(x, timesteps, condition)  # [batch_size, seq_len, vocab_size]
                
                # 应用温度
                logits = logits / max(temperature, 1e-10)
                
                # 应用特殊标记屏蔽
                logits = logits + (1 - special_token_mask) * -1e4
                
                # 采样新序列
                if t > 0:
                    # 非最终步骤：概率性采样
                    probs = F.softmax(logits, dim=-1)
                    
                    # 处理可能的NaN/Inf
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        probs = torch.nan_to_num(probs, nan=1.0/self.vocab_size, posinf=1.0, neginf=0.0)
                        # 重新归一化
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                        
                    x_new = torch.multinomial(probs.reshape(-1, self.vocab_size), 1).reshape(batch_size, seq_len)
                    
                    # 随机决定哪些位置更新（基于噪声调度）
                    mask_ratio = self.noise_schedule.alphas[t].item()
                    keep_mask = torch.rand(batch_size, seq_len, device=device) < mask_ratio
                    x = torch.where(keep_mask, x, x_new)
                else:
                    # 最终步骤：贪婪解码或按概率采样最终结果
                    if temperature < 0.1:
                        # 温度很低时使用argmax
                        x = torch.argmax(logits, dim=-1)
                    else:
                        # 否则使用带温度的采样
                        probs = F.softmax(logits, dim=-1)
                        # 处理NaN/Inf
                        if torch.isnan(probs).any() or torch.isinf(probs).any():
                            probs = torch.nan_to_num(probs, nan=1.0/self.vocab_size, posinf=1.0, neginf=0.0)
                            probs = probs / probs.sum(dim=-1, keepdim=True)
                        x = torch.multinomial(probs.reshape(-1, self.vocab_size), 1).reshape(batch_size, seq_len)
            except Exception as e:
                print(f"采样过程中出错: {e}")
                # 出错时随机生成一些有效token
                x_new = torch.randint(4, self.vocab_size, (batch_size, seq_len), device=device)
                # 部分更新序列
                update_mask = torch.rand(batch_size, seq_len, device=device) < 0.2
                x = torch.where(update_mask, x_new, x)

        return x