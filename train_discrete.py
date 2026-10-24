import os
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import traceback

from data.tokenizer import ProteinTokenizer
from data.dataset import AMPPreferenceDataset
from models.discrete_diffusion import DiscreteProteinDiffusionModel
from models.discrete_dpo import DiscreteAMP_DPO

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('discrete_amp_dpo_train.log')
    ]
)
logger = logging.getLogger('discrete_amp_dpo')

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_train_args():
    """获取训练参数"""
    parser = argparse.ArgumentParser(description='离散抗菌肽扩散模型训练')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='data/amp_preference_train.csv',
                        help='训练数据路径')
    parser.add_argument('--val_data_path', type=str, default='data/amp_preference_val.csv',
                        help='验证数据路径')
    parser.add_argument('--max_length', type=int, default=50,
                        help='最大序列长度')

    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='隐藏层维度')
    parser.add_argument('--condition_dim', type=int, default=32,
                        help='条件嵌入维度')
    parser.add_argument('--transformer_layers', type=int, default=6,
                        help='Transformer层数')
    parser.add_argument('--transformer_heads', type=int, default=8,
                        help='Transformer注意力头数')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='扩散步骤数量')
    parser.add_argument('--beta_schedule', type=str, default='linear',
                        help='Beta调度类型 (linear, cosine, sqrt)')

    # 预训练模型
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='预训练模型路径')

    # DPO参数
    parser.add_argument('--beta', type=float, default=10.0,  # 修改默认beta值为10.0
                        help='DPO的beta参数')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,  # 降低默认学习率以提高稳定性
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/discrete_dpo',
                        help='检查点保存目录')
    parser.add_argument('--save_every', type=int, default=5,
                        help='每多少轮保存一次模型')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停耐心值')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的检查点路径')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,  # 降低梯度裁剪阈值以提高稳定性
                        help='梯度裁剪最大范数')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')

    # 随机种子
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 调试模式
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    
    # 渐进式beta增长
    parser.add_argument('--progressive_beta', action='store_true', 
                        help='启用渐进式beta增长')
    parser.add_argument('--initial_beta', type=float, default=1.0,
                        help='初始beta值（渐进式增长时使用）')

    return parser.parse_args()

def validate(dpo_trainer, val_loader, device):
    """在验证集上评估模型"""
    logger.info("开始验证...")
    dpo_trainer.model.eval()  # 设置为评估模式
    total_loss = 0
    valid_batches = 0

    with torch.no_grad():  # 不计算梯度
        for batch in tqdm(val_loader, desc="验证中"):
            try:
                # 将数据移至设备
                win_seq = batch['win_seq'].to(device)
                lose_seq = batch['lose_seq'].to(device)

                conditions = None
                if 'properties' in batch:
                    conditions = batch['properties'].to(device)

                # 计算损失
                loss = dpo_trainer.compute_loss(win_seq, lose_seq, conditions)
                
                # 只记录有效的损失值
                if isinstance(loss, torch.Tensor) and loss.item() != 1.0 and not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    valid_batches += 1
            except Exception as e:
                logger.error(f"验证批次出错: {e}")
                logger.error(traceback.format_exc())
                continue

    # 确保至少有一个有效批次
    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
    else:
        logger.warning("验证中没有有效批次，返回默认损失值")
        avg_loss = 999.9  # 一个明显的大值，表示验证失败
        
    dpo_trainer.model.train()  # 恢复训练模式
    return avg_loss

def load_pretrained_model(model, pretrained_path, device):
    """加载预训练模型权重"""
    if not os.path.exists(pretrained_path):
        logger.warning(f"预训练模型路径不存在: {pretrained_path}")
        return model
    
    logger.info(f"加载预训练模型: {pretrained_path}")
    try:
        pretrained = torch.load(pretrained_path, map_location=device)
        
        # 输出预训练模型配置
        if 'config' in pretrained:
            logger.info(f"预训练模型配置: {pretrained['config']}")
        
        # 加载预训练模型权重
        if 'model_state_dict' in pretrained:
            pretrained_dict = pretrained['model_state_dict']
            model_dict = model.state_dict()
            
            # 获取模型权重形状统计信息
            compatible_count = 0
            incompatible_count = 0
            
            # 打印所有层的形状差异
            for key in pretrained_dict.keys():
                if key in model_dict:
                    if pretrained_dict[key].shape == model_dict[key].shape:
                        compatible_count += 1
                    else:
                        incompatible_count += 1
                        logger.warning(f"层 {key} 形状不匹配 - 预训练: {pretrained_dict[key].shape}, 当前: {model_dict[key].shape}")
            
            logger.info(f"兼容层: {compatible_count}, 不兼容层: {incompatible_count}")
            
            # 精确匹配
            compatible_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            # 特殊处理：尝试加载关键层
            key_layers = ['token_embedding.weight', 'position_embedding.weight', 'time_mlp']
            for key in key_layers:
                if key in pretrained_dict and key in model_dict:
                    if pretrained_dict[key].shape == model_dict[key].shape:
                        compatible_dict[key] = pretrained_dict[key]
                        logger.info(f"已加载关键层: {key}")
            
            # 检查加载比例
            if len(compatible_dict) / len(model_dict) < 0.8:
                logger.warning(f"加载的权重比例较低: {len(compatible_dict)}/{len(model_dict)}，可能会影响性能")
            
            # 更新模型权重
            if compatible_dict:
                model_dict.update(compatible_dict)
                model.load_state_dict(model_dict)
                logger.info(f"成功加载了 {len(compatible_dict)}/{len(model_dict)} 层预训练权重")
            else:
                logger.error("没有找到兼容的权重层！使用随机初始化模型")
        else:
            logger.warning("预训练模型中没有找到model_state_dict")
    except Exception as e:
        logger.error(f"加载预训练模型时出错: {e}")
        logger.error(traceback.format_exc())
    
    return model

def main():
    # 获取参数
    args = get_train_args()
    
    # 安全检查和参数修正
    if args.beta > 100:
        logger.warning(f"Beta值 ({args.beta}) 过高，自动降低到10以提高稳定性")
        args.beta = 10.0
    
    if args.learning_rate > 1e-3:
        logger.warning(f"学习率 ({args.learning_rate}) 过高，自动降低到1e-4以提高稳定性")
        args.learning_rate = 1e-4
    
    # 设置随机种子
    set_seed(args.seed)

    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 初始化分词器
    tokenizer = ProteinTokenizer()

    # 创建数据集和数据加载器
    train_dataset = AMPPreferenceDataset(args.data_path, tokenizer, max_length=args.max_length)
    val_dataset = AMPPreferenceDataset(args.val_data_path, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    logger.info(f"训练数据集大小: {len(train_dataset)}")
    logger.info(f"验证数据集大小: {len(val_dataset)}")

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
        debug=args.debug  # 是否启用调试模式
    )

    # 设置设备
    device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    # 加载预训练模型（如果提供）
    if args.pretrained_model:
        model = load_pretrained_model(model, args.pretrained_model, device)

    # 创建参考模型
    ref_model = None  # 将在DPO初始化时创建

    # 确定初始beta值
    initial_beta = args.initial_beta if args.progressive_beta else args.beta
    logger.info(f"初始beta值: {initial_beta}, 目标beta值: {args.beta}, 渐进式增长: {args.progressive_beta}")

    # 初始化DPO训练器
    dpo_trainer = DiscreteAMP_DPO(model, ref_model, beta=initial_beta)

    # 恢复训练（如果提供检查点）
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"从检查点恢复训练: {args.resume}")
            try:
                checkpoint = torch.load(args.resume, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                logger.info(f"从轮次 {start_epoch} 继续训练")
            except Exception as e:
                logger.error(f"加载检查点时出错: {e}")
                logger.error(traceback.format_exc())
                start_epoch = 0
        else:
            logger.warning(f"检查点路径不存在: {args.resume}")

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8  # 增加eps参数提高数值稳定性
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,  # 更平缓的学习率衰减
        patience=2,   # 更早触发学习率降低
        verbose=True,
        min_lr=1e-6   # 设置最小学习率
    )

    # 移至设备
    model.to(device)
    dpo_trainer.ref_model.to(device)

    # 初始化早停参数
    best_val_loss = float('inf')
    patience_counter = 0

    # 训练历史记录
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'learning_rates': [], 
        'beta_values': []
    }

    logger.info(f"开始训练，使用设备: {device}")

    for epoch in range(start_epoch, args.num_epochs):
        # 如果使用渐进式beta增长，计算当前epoch的beta值
        if args.progressive_beta:
            # 计算当前epoch的beta值（线性增长）
            progress = min(1.0, epoch / (args.num_epochs * 0.5))  # 在一半的epoch内达到目标beta
            current_beta = initial_beta + (args.beta - initial_beta) * progress
            dpo_trainer.beta = current_beta
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}, 当前beta值: {current_beta:.4f}")
        else:
            current_beta = args.beta
        
        # 记录当前beta值
        history['beta_values'].append(current_beta)
        
        # 训练一个epoch
        epoch_loss = 0
        valid_batches = 0
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
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
                loss = dpo_trainer.train_step(new_batch, optimizer)

                # 只记录有效的损失值
                if loss != 1.0 and not np.isnan(loss) and not np.isinf(loss):
                    epoch_loss += loss
                    valid_batches += 1
                else:
                    logger.warning(f"批次 {batch_idx}: 跳过无效损失值: {loss}")

                # 更新进度条
                pbar.set_postfix(loss=loss, lr=current_lr, beta=current_beta)
                
                # 每100个批次输出一次状态
                if batch_idx % 100 == 0 and batch_idx > 0:
                    if valid_batches > 0:
                        avg_train_loss_so_far = epoch_loss / valid_batches
                        logger.info(f"Batch {batch_idx}/{len(train_loader)}, 平均损失: {avg_train_loss_so_far:.6f}")
                
            except Exception as e:
                logger.error(f"训练批次 {batch_idx} 出错: {e}")
                logger.error(traceback.format_exc())
                continue

        # 计算平均训练损失
        if valid_batches > 0:
            avg_train_loss = epoch_loss / valid_batches
        else:
            logger.warning("本轮没有有效批次")
            avg_train_loss = 999.9  # 一个明显的大值表示训练失败
            
        history['train_loss'].append(avg_train_loss)

        # 在验证集上评估
        val_loss = validate(dpo_trainer, val_loader, device)
        history['val_loss'].append(val_loss)

        # 更新学习率调度器
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1}/{args.num_epochs}: 训练损失={avg_train_loss:.6f}, 验证损失={val_loss:.6f}, LR={current_lr}, Beta={current_beta:.4f}")

        # 保存常规检查点
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"discrete_amp_dpo_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'tokenizer': tokenizer,
                'args': vars(args),
                'beta': current_beta,
                'history': history
            }, checkpoint_path)
            logger.info(f"保存检查点到 {checkpoint_path}")

        # 检查是否需要保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, "discrete_amp_dpo_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'tokenizer': tokenizer,
                'args': vars(args),
                'beta': current_beta,
                'history': history
            }, best_checkpoint_path)
            logger.info(f"保存最佳模型到 {best_checkpoint_path}, 验证损失: {val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"验证损失未改善, 耐心计数: {patience_counter}/{args.patience}")

        # 检查是否应该早停
        if patience_counter >= args.patience:
            logger.info(f"验证损失已连续{args.patience}个epoch未改善，早停")
            break

        # 每个epoch后保存训练历史
        history_path = os.path.join(args.checkpoint_dir, "training_history.json")
        with open(history_path, 'w') as f:
            # 将NumPy数组转换为列表以便JSON序列化
            serializable_history = {
                'train_loss': [float(x) for x in history['train_loss']],
                'val_loss': [float(x) for x in history['val_loss']],
                'learning_rates': [float(x) for x in history['learning_rates']],
                'beta_values': [float(x) for x in history['beta_values']]
            }
            json.dump(serializable_history, f, indent=2)

    # 保存最终模型
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "discrete_amp_dpo_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'args': vars(args),
        'history': history
    }, final_checkpoint_path)

    logger.info(f"训练完成，最终模型已保存到 {final_checkpoint_path}")

    # 绘制训练历史
    plt.figure(figsize=(12, 8))
    
    # 绘制损失
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Discrete DPO Training History')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制学习率和Beta值
    plt.subplot(2, 1, 2)
    plt.plot(history['learning_rates'], label='Learning Rate', color='green')
    plt.plot(history['beta_values'], label='Beta Value', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Learning Rate and Beta Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()

    # 保存图表
    history_plot_path = os.path.join(args.checkpoint_dir, "discrete_training_history.png")
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"训练历史图表已保存到 {history_plot_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"训练过程中发生未处理的异常: {e}")
        logger.error(traceback.format_exc())
        raise