import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm
import numpy as np

class DiscreteAMP_DPO:
    """基于直接偏好优化的离散抗菌肽扩散模型"""
    
    def __init__(self, model, ref_model=None, beta=10.0):
        """
        初始化DPO训练器
        
        Args:
            model: 要优化的离散扩散模型
            ref_model: 参考模型（如果为None，则从model创建副本）
            beta: KL散度惩罚系数
        """
        self.model = model
        
        # 如果没有参考模型，则创建一个模型副本
        if ref_model is None:
            self.ref_model = copy.deepcopy(model)
        else:
            self.ref_model = ref_model
            
        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # 将beta值从默认的5000降低到10
        self.beta = beta
        self.noise_schedule = model.noise_schedule
    
    def compute_loss(self, win_seq, lose_seq, conditions=None):
        """计算离散DPO损失"""
        batch_size = win_seq.shape[0]
        device = win_seq.device
        
        try:
            # 随机采样时间步
            t = torch.randint(0, self.noise_schedule.num_timesteps, (batch_size,), device=device)
            
            # 对胜利和失败序列添加相同的噪声模式
            # 首先为win_seq生成噪声掩码
            noisy_win_seq, noise_mask = self.noise_schedule.q_sample(win_seq, t)
            
            # 使用相同的噪声掩码处理lose_seq
            noisy_lose_seq = lose_seq.clone()
            noisy_lose_seq[noise_mask] = self.noise_schedule.mask_id
            
            # 如果噪声掩码全为空，添加一些随机掩码
            if noise_mask.sum() == 0:
                random_mask = torch.rand_like(win_seq.float(), device=device) < 0.1
                noisy_win_seq[random_mask] = self.noise_schedule.mask_id
                noisy_lose_seq[random_mask] = self.noise_schedule.mask_id
                noise_mask = random_mask
            
            # 使用当前模型预测logits
            win_logits = self.model(noisy_win_seq, t, conditions)  # [batch_size, seq_len, vocab_size]
            lose_logits = self.model(noisy_lose_seq, t, conditions)  # [batch_size, seq_len, vocab_size]
            
            # 检查是否有NaN值
            if torch.isnan(win_logits).any() or torch.isnan(lose_logits).any():
                print("警告: DPO损失计算中检测到NaN logits")
                return torch.tensor(1.0, device=device, requires_grad=True)
            
            # 使用参考模型预测logits
            with torch.no_grad():
                ref_win_logits = self.ref_model(noisy_win_seq, t, conditions)
                ref_lose_logits = self.ref_model(noisy_lose_seq, t, conditions)
            
            # 仅在被噪化的位置计算分布差异
            win_log_probs = F.log_softmax(win_logits, dim=-1)
            lose_log_probs = F.log_softmax(lose_logits, dim=-1)
            ref_win_log_probs = F.log_softmax(ref_win_logits, dim=-1)
            ref_lose_log_probs = F.log_softmax(ref_lose_logits, dim=-1)
            
            # 获取目标token的log概率
            win_token_log_probs = torch.gather(win_log_probs.reshape(-1, self.model.vocab_size), 
                                             1, 
                                             win_seq.reshape(-1, 1)).reshape(batch_size, -1)
            lose_token_log_probs = torch.gather(lose_log_probs.reshape(-1, self.model.vocab_size), 
                                              1, 
                                              lose_seq.reshape(-1, 1)).reshape(batch_size, -1)
            
            ref_win_token_log_probs = torch.gather(ref_win_log_probs.reshape(-1, self.model.vocab_size), 
                                                 1, 
                                                 win_seq.reshape(-1, 1)).reshape(batch_size, -1)
            ref_lose_token_log_probs = torch.gather(ref_lose_log_probs.reshape(-1, self.model.vocab_size), 
                                                  1, 
                                                  lose_seq.reshape(-1, 1)).reshape(batch_size, -1)
            
            # 只关注被噪化的位置
            win_token_log_probs = win_token_log_probs * noise_mask.float()
            lose_token_log_probs = lose_token_log_probs * noise_mask.float()
            ref_win_token_log_probs = ref_win_token_log_probs * noise_mask.float()
            ref_lose_token_log_probs = ref_lose_token_log_probs * noise_mask.float()
            
            # 计算每个序列的总对数概率
            win_log_prob = win_token_log_probs.sum(dim=1)
            lose_log_prob = lose_token_log_probs.sum(dim=1)
            ref_win_log_prob = ref_win_token_log_probs.sum(dim=1)
            ref_lose_log_prob = ref_lose_token_log_probs.sum(dim=1)
            
            # 计算π和π_ref策略之间的对数概率比
            win_logratios = win_log_prob - ref_win_log_prob
            lose_logratios = lose_log_prob - ref_lose_log_prob
            
            # 添加调试日志 - 输出比率统计信息
            win_ratio_mean = win_logratios.mean().item()
            lose_ratio_mean = lose_logratios.mean().item()
            ratio_diff = win_ratio_mean - lose_ratio_mean
            # print(f"比率统计 - Win: {win_ratio_mean:.4f}, Lose: {lose_ratio_mean:.4f}, 差异: {ratio_diff:.4f}")
            
            # 检查比率的有效性
            if torch.isnan(win_logratios).any() or torch.isnan(lose_logratios).any():
                print("警告: DPO损失计算中检测到NaN比率")
                return torch.tensor(1.0, device=device, requires_grad=True)
            
            # 应用比率范围限制 - 降低限制范围增加稳定性
            win_logratios = torch.clamp(win_logratios, -20.0, 20.0)
            lose_logratios = torch.clamp(lose_logratios, -20.0, 20.0)
            
            # 计算DPO损失
            inside_term = self.beta * (win_logratios - lose_logratios)
            unclamped_loss = -F.logsigmoid(inside_term).mean()
            
            # 添加调试信息 - 显示内部项和未截断损失
            # print(f"DPO内部项: {inside_term.mean().item():.4f}, 未截断损失: {unclamped_loss.item():.4f}, beta: {self.beta:.4f}")
            
            # 检查最终损失的有效性
            if torch.isnan(unclamped_loss) or torch.isinf(unclamped_loss):
                print(f"警告: DPO最终损失无效: {unclamped_loss.item()}")
                return torch.tensor(1.0, device=device, requires_grad=True)
            
            # 应用损失范围限制 - 降低上限从100降到20
            loss = torch.clamp(unclamped_loss, 0.0, 20.0)
            
            return loss
            
        except Exception as e:
            print(f"计算DPO损失时出错: {e}")
            return torch.tensor(1.0, device=device, requires_grad=True)
    
    def train_step(self, batch, optimizer):
        """
        执行一步训练
        
        Args:
            batch: 包含win_seq和lose_seq的批次
            optimizer: 优化器
            
        Returns:
            损失值
        """
        try:
            # 提取批次数据
            win_seq = batch['win_seq']
            lose_seq = batch['lose_seq']
            conditions = batch.get('properties', None)
            
            # 计算损失
            loss = self.compute_loss(win_seq, lose_seq, conditions)
            
            # 检查损失值是否合理 - 添加额外检查
            if loss.item() > 19.9:  # 接近最大截断值
                print(f"警告: 损失值非常高 ({loss.item():.4f})，可能表明训练不稳定")
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 增强梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 监控梯度范数
            # if grad_norm > 0.9:  # 接近最大裁剪值
            #     print(f"警告: 高梯度范数 ({grad_norm:.4f})，可能需要降低学习率")
            
            # 检查梯度是否包含NaN
            valid_grad = True
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        valid_grad = False
                        print(f"警告: 参数 {name} 中检测到NaN/Inf梯度")
                        break
            
            # 仅在梯度有效时更新
            if valid_grad:
                optimizer.step()
            else:
                print("跳过参数更新，因为检测到无效梯度")
            
            return loss.item()
        except Exception as e:
            print(f"训练步骤出错: {e}")
            return 1.0  # 返回一个默认损失值
    
    def train(self, dataloader, optimizer, num_epochs, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        训练模型
        
        Args:
            dataloader: 数据加载器
            optimizer: 优化器
            num_epochs: 训练轮数
            device: 计算设备
            
        Returns:
            训练历史记录
        """
        # 将模型移至设备
        self.model.to(device)
        self.ref_model.to(device)
        
        history = {'loss': []}
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            valid_batches = 0
            
            # 创建进度条
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                try:
                    # 将数据移至设备
                    win_seq = batch['win_seq'].to(device)
                    lose_seq = batch['lose_seq'].to(device)
                    
                    conditions = None
                    if 'properties' in batch:
                        conditions = batch['properties'].to(device)
                    
                    # 创建新的批次
                    new_batch = {
                        'win_seq': win_seq,
                        'lose_seq': lose_seq
                    }
                    
                    if conditions is not None:
                        new_batch['properties'] = conditions
                    
                    # 执行训练步骤
                    loss = self.train_step(new_batch, optimizer)
                    
                    # 只记录有效的损失值
                    if loss != 1.0 and not np.isnan(loss) and not np.isinf(loss):
                        epoch_loss += loss
                        valid_batches += 1
                    
                    # 更新进度条
                    pbar.set_postfix(loss=loss)
                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    continue
            
            # 计算平均损失
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
            else:
                print("警告: 本轮没有有效批次")
                avg_loss = 999.9  # 一个明显的大值表示训练失败
                
            history['loss'].append(avg_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        return history
    
    def evaluate(self, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            device: 计算设备
            
        Returns:
            平均损失
        """
        self.model.eval()
        self.ref_model.eval()
        
        total_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                try:
                    # 将数据移至设备
                    win_seq = batch['win_seq'].to(device)
                    lose_seq = batch['lose_seq'].to(device)
                    
                    conditions = None
                    if 'properties' in batch:
                        conditions = batch['properties'].to(device)
                    
                    # 计算损失
                    loss = self.compute_loss(win_seq, lose_seq, conditions)
                    
                    # 只记录有效的损失值
                    if loss.item() != 1.0 and not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item()
                        valid_batches += 1
                except Exception as e:
                    print(f"验证批次出错: {e}")
                    continue
        
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
        else:
            print("警告: 验证中没有有效批次")
            avg_loss = 999.9  # 一个明显的大值表示验证失败
            
        print(f"Evaluation Loss: {avg_loss:.6f}")
        
        self.model.train()
        return avg_loss