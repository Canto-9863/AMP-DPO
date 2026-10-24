import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
import random
import logging
from datetime import datetime

from data.tokenizer import ProteinTokenizer
from models.discrete_diffusion import DiscreteProteinDiffusionModel

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('discrete_protein_pretrain.log')
    ]
)
logger = logging.getLogger('discrete_protein_pretrain')

class ProteinDataset(Dataset):
    """蛋白质序列数据集"""
    def __init__(self, data_dir, tokenizer, max_length=100, use_index=True):
        """
        初始化蛋白质数据集
        
        Args:
            data_dir: 数据目录，包含批次CSV文件
            tokenizer: 分词器，用于序列标记化
            max_length: 最大序列长度
            use_index: 是否使用索引文件
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载批次文件
        if use_index and (self.data_dir / 'index.csv').exists():
            self.batch_files = pd.read_csv(self.data_dir / 'index.csv')['file'].tolist()
        else:
            self.batch_files = [f for f in os.listdir(self.data_dir)
                               if f.endswith('.csv') and f != 'index.csv']

        logger.info(f"找到 {len(self.batch_files)} 个批次文件")

        # 加载第一个批次以估计数据集大小
        if self.batch_files:
            first_batch = pd.read_csv(self.data_dir / self.batch_files[0])
            self.estimated_size = len(first_batch) * len(self.batch_files)
            logger.info(f"估计数据集大小: {self.estimated_size}")
        else:
            self.estimated_size = 0
            logger.warning(f"未找到批次文件!")

    def __len__(self):
        return self.estimated_size

    def __getitem__(self, idx):
        """获取样本"""
        # 确定批次索引和样本索引
        batch_idx = idx % len(self.batch_files)
        batch_file = self.batch_files[batch_idx]

        # 加载批次
        batch_df = pd.read_csv(self.data_dir / batch_file)

        # 确保不会越界
        sample_idx = (idx // len(self.batch_files)) % len(batch_df)

        # 获取序列
        sequence = batch_df.iloc[sample_idx]['sequence']

        # 标记化
        tokens = self.tokenizer.encode(sequence, max_length=self.max_length)

        return {'sequence': tokens}

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pretrain_args():
    """获取预训练参数"""
    parser = argparse.ArgumentParser(description='离散扩散模型预训练')

    # 数据参数
    parser.add_argument('--train_data', type=str, required=True, help='训练数据目录')
    parser.add_argument('--val_data',
                        type=str,
                        default=None,
                        help='验证数据目录 (可选)')
    parser.add_argument('--max_length', type=int, default=100, help='最大序列长度')

    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--transformer_layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--transformer_heads', type=int, default=8, help='Transformer注意力头数')
    parser.add_argument('--num_timesteps',
                        type=int,
                        default=1000,
                        help='扩散步骤数量')
    parser.add_argument('--beta_schedule',
                        type=str,
                        default='linear',
                        help='Beta调度类型 (linear, cosine, sqrt)')
    parser.add_argument('--condition_dim', type=int, default=32,
                        help='条件嵌入维度')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-6,
                        help='权重衰减')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='checkpoints/discrete_pretrain',
                        help='检查点保存目录')
    parser.add_argument('--save_every', type=int, default=1, help='每多少轮保存一次模型')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='数据加载器工作进程数')
    parser.add_argument('--eval_steps',
                        type=int,
                        default=1000,
                        help='每多少步进行一次验证')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=1.0,
                        help='梯度裁剪最大范数')

    # 设备参数
    parser.add_argument('--device',
                        type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')
    parser.add_argument('--precision',
                        type=str,
                        default='float32',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='训练精度')

    # 随机种子
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    return parser.parse_args()

def validate(model, val_loader, device, num_batches=None):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0
    batches = 0
    valid_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中", total=num_batches):
            try:
                # 获取序列
                x = batch['sequence'].to(device)

                # 计算预训练损失
                loss = model.compute_loss(x)
                
                # 只累加有效的损失值
                if isinstance(loss, torch.Tensor) and loss.item() > 0 and not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    valid_batches += 1
                
                batches += 1

                if num_batches is not None and batches >= num_batches:
                    break
                    
            except Exception as e:
                logger.error(f"处理验证批次时出错: {e}")
                continue

    # 确保至少有一个有效批次
    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
    else:
        logger.warning("验证中没有有效批次，返回默认损失值")
        avg_loss = 999.9  # 一个明显的大值，表示验证失败
        
    model.train()
    return avg_loss

def get_optimizer(model, args):
    """获取优化器"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

def get_scheduler(optimizer, args, total_steps):
    """获取学习率调度器"""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.learning_rate / 100
    )

def save_checkpoint(model, optimizer, scheduler, epoch, step, args, loss, val_loss=None, is_best=False):
    """保存检查点"""
    checkpoint_path = Path(args.checkpoint_dir) / f"discrete_diffusion_pretrained_e{epoch}_s{step}.pt"

    # 保存模型状态
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'val_loss': val_loss,
        'args': vars(args)
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"模型保存到 {checkpoint_path}")

    # 如果是最佳模型，另存一份
    if is_best:
        best_path = Path(args.checkpoint_dir) / "discrete_diffusion_pretrained_best.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"最佳模型保存到 {best_path}")

    # 最新模型
    latest_path = Path(args.checkpoint_dir) / "discrete_diffusion_pretrained_latest.pt"
    torch.save(checkpoint, latest_path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """加载检查点"""
    logger.info(f"从 {path} 加载检查点")
    checkpoint = torch.load(path, map_location='cpu')

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 返回轮数和步骤
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)

    return model, optimizer, scheduler, epoch, step

def plot_training_curve(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss')

    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"训练曲线已保存到 {save_path}")

def main():
    args = get_pretrain_args()

    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    # 初始化分词器
    tokenizer = ProteinTokenizer()

    # 加载数据集
    train_dataset = ProteinDataset(args.train_data, tokenizer, max_length=args.max_length)
    logger.info(f"训练数据集大小: {len(train_dataset)}")

    # 设置验证集
    val_dataset = None
    if args.val_data:
        val_dataset = ProteinDataset(args.val_data, tokenizer, max_length=args.max_length)
        logger.info(f"验证数据集大小: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # 初始化模型
    model = DiscreteProteinDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        condition_dim=args.condition_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        conditional=True,
        pad_id=tokenizer.pad_id,
        debug=False)

    # 移至设备
    model.to(device)

    # 设置精度
    if args.precision == 'float16' and device.type == 'cuda':
        model.half()
        logger.info("使用半精度 (float16) 训练")
    elif args.precision == 'bfloat16' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model.to(dtype=torch.bfloat16)
        logger.info("使用bfloat16精度训练")
    else:
        logger.info("使用全精度 (float32) 训练")

    # 优化器和调度器
    optimizer = get_optimizer(model, args)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_scheduler(optimizer, args, total_steps)

    start_epoch = 0
    start_step = 0
    best_val_loss = float('inf')

    if args.resume:
        model, optimizer, scheduler, start_epoch, start_step = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        logger.info(f"恢复训练，从轮 {start_epoch} 步骤 {start_step} 开始")

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }

    # 验证频率
    eval_steps = min(args.eval_steps, len(train_loader))

    # 训练循环
    logger.info("开始训练...")
    steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        valid_steps = 0

        # 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for i, batch in enumerate(pbar):
            global_step = epoch * steps_per_epoch + i + 1

            if global_step <= start_step:
                continue

            try:
                # 获取序列
                x = batch['sequence'].to(device)

                # 计算损失
                loss = model.compute_loss(x)
                
                # 确保损失是有效的
                if isinstance(loss, torch.Tensor) and not torch.isnan(loss) and not torch.isinf(loss):
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪 - 防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # 更新参数
                    optimizer.step()
                    scheduler.step()
                    
                    # 记录有效损失
                    epoch_loss += loss.item()
                    valid_steps += 1
                else:
                    logger.warning(f"跳过无效损失: {loss}")
                    
                step += 1

                # 更新进度条
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss.item():.6f}" if isinstance(loss, torch.Tensor) else "N/A", 
                                lr=f"{current_lr:.8f}")

                # 验证
                if val_loader is not None and global_step % eval_steps == 0:
                    val_loss = validate(model, val_loader, device, num_batches=100)
                    logger.info(f"Step {global_step}, 验证损失: {val_loss:.6f}")

                    # 检查是否是最佳模型
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, optimizer, scheduler,
                            epoch, global_step, args,
                            loss.item() if isinstance(loss, torch.Tensor) else 0.0, 
                            val_loss, is_best=True
                        )
            except RuntimeError as e:
                logger.error(f"训练步骤出错: {e}")
                continue

        # 计算平均损失
        if valid_steps > 0:
            avg_loss = epoch_loss / valid_steps
        else:
            logger.warning("本轮没有有效步骤，跳过损失计算")
            avg_loss = 999.9  # 使用一个明显的大值表示训练失败
            
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, 平均损失: {avg_loss:.6f}")

        # 更新历史记录
        history['train_loss'].append(avg_loss)
        history['learning_rates'].append(scheduler.get_last_lr()[0])

        # 完整验证
        if val_loader is not None:
            val_loss = validate(model, val_loader, device)
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}, 验证损失: {val_loss:.6f}")
            history['val_loss'].append(val_loss)

            # 检查是否是最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            is_best = False

        # 保存检查点
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            save_checkpoint(
                model, optimizer, scheduler,
                epoch + 1, global_step, args,
                avg_loss, val_loss if val_loader else None,
                is_best=is_best
            )

        # 绘制训练曲线
        plot_training_curve(
            history['train_loss'],
            history['val_loss'] if val_loader else None,
            os.path.join(args.checkpoint_dir, "discrete_training_curve.png")
        )

        # 保存训练历史
        with open(os.path.join(args.checkpoint_dir, "discrete_training_history.json"), 'w') as f:
            json.dump(history, f, indent=4)

    # 保存最终模型
    final_path = os.path.join(args.checkpoint_dir, "discrete_diffusion_pretrained_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': tokenizer.vocab_size,
            'transformer_layers': args.transformer_layers,
            'transformer_heads': args.transformer_heads
        }
    }, final_path)

    logger.info(f"训练完成! 最终模型保存到 {final_path}")

if __name__ == "__main__":
    main()