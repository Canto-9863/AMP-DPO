import torch
import numpy as np

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
        
        # 确保t在有效范围内
        t = torch.clamp(t, 0, self.num_timesteps - 1)
        
        # 获取t时刻的累积腐蚀概率
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_0.shape)
        
        # 确保腐蚀概率不会太小
        # 限制最大值为0.95，确保至少有5%的位置被掩码化
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
        
        # 提取值
        out = a.gather(-1, t.cpu()).to(t.device)
        
        if shape is None:
            return out
        
        # 调整形状以便于广播
        reshape_dim = [batch_size] + [1] * (len(shape) - 1)
        return out.reshape(reshape_dim)