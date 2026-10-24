import argparse
import torch


def get_train_args():
    """获取训练参数"""
    parser = argparse.ArgumentParser(description='抗菌肽扩散模型训练')

    # 数据参数
    parser.add_argument('--data_path',
                        type=str,
                        default='data/sample_amp_data_train.csv',
                        help='训练数据路径')
    parser.add_argument('--val_data_path',
                        type=str,
                        default='data/sample_amp_data_val.csv',
                        help='验证数据路径')
    parser.add_argument('--max_length', type=int, default=50, help='最大序列长度')

    # 模型参数 - 确保维度兼容
    parser.add_argument('--embedding_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--unet_dim', type=int, default=128, help='UNet基础维度')
    parser.add_argument('--condition_dim', type=int, default=32, help='条件嵌入维度')
    parser.add_argument('--num_timesteps',
                        type=int,
                        default=1000,
                        help='扩散步骤数量')
    parser.add_argument('--beta_schedule',
                        type=str,
                        default='linear',
                        help='Beta调度类型 (linear, cosine, sqrt)')

    # DPO参数
    parser.add_argument('--beta', type=float, default=5000, help='DPO的beta参数')

    # 训练参数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,  # 减小批量大小
        help='批次大小')
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,  # 减少训练轮数
        help='训练轮数')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-4,
                        help='权重衰减')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--save_every', type=int, default=5, help='每多少轮保存一次模型')

    # 设备参数
    parser.add_argument('--device',
                        type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')

    # 随机种子
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    parser.add_argument('--pretrained_model',
                        type=str,
                        default='',
                        help='预训练模型路径')
    return parser.parse_args()
def get_generate_args():
    """获取生成参数"""
    parser = argparse.ArgumentParser(description='抗菌肽生成')

    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')

    # 生成参数
    parser.add_argument('--num_samples', type=int, default=50,
                        help='生成的样本数量')
    parser.add_argument('--seq_len', type=int, default=30,
                        help='生成的序列长度')
    parser.add_argument('--output_file', type=str, default='generated_amps.csv',
                        help='生成结果输出文件')

    # 条件参数
    parser.add_argument('--charge', type=float, default=2.0,
                        help='目标电荷')
    parser.add_argument('--hydrophobicity', type=float, default=0.5,
                        help='目标疏水性')
    parser.add_argument('--length', type=float, default=20.0,
                        help='目标长度')

    # 评估参数
    parser.add_argument('--top_k', type=int, default=10,
                        help='选择的顶级候选数量')
    parser.add_argument('--min_amp_score', type=float, default=0.6,
                        help='最小抗菌活性分数')
    parser.add_argument('--max_toxicity', type=float, default=0.4,
                        help='最大毒性分数')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备')

    return parser.parse_args()
