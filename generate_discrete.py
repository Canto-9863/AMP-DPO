import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import argparse
import logging
from collections import defaultdict

from data.tokenizer import ProteinTokenizer
from models.discrete_diffusion import DiscreteProteinDiffusionModel
from utils.evaluation import AMPEvaluator
from utils.protein_utils import evaluate_amp_properties

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec

# 设置中文字体
try:
    # Windows 系统
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'DejaVu Sans', 'Arial']
    else:
        # 尝试其他常见中文字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Arial']

    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("Warning: Unable to set Chinese font, Chinese characters in the chart may not display correctly")


# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('discrete_amp_generate.log')
    ]
)
logger = logging.getLogger('discrete_amp_generate')

def get_generate_args():
    """获取生成参数"""
    parser = argparse.ArgumentParser(description='离散抗菌肽生成')

    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')

    # 生成参数
    parser.add_argument('--num_samples', type=int, default=100,
                        help='总生成样本数量')
    parser.add_argument('--seq_len_min', type=int, default=15,
                        help='最小序列长度')
    parser.add_argument('--seq_len_max', type=int, default=30,
                        help='最大序列长度')
    parser.add_argument('--output_dir', type=str, default='generated_results',
                        help='生成结果输出目录')
    parser.add_argument('--output_prefix', type=str, default='discrete_amps',
                        help='输出文件前缀')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='生成批次大小')
    parser.add_argument('--num_strategies', type=int, default=5,
                        help='使用的策略数量')
    parser.add_argument('--iterations', type=int, default=3,
                        help='迭代优化次数')

    # 评估参数
    parser.add_argument('--top_k', type=int, default=20,
                        help='选择的顶级候选数量')
    parser.add_argument('--min_amp_score', type=float, default=0.6,
                        help='最小抗菌活性分数')
    parser.add_argument('--max_toxicity', type=float, default=0.4,
                        help='最大毒性分数')
    parser.add_argument('--min_stability', type=float, default=0.5,
                        help='最小稳定性分数')
    parser.add_argument('--min_amphipathicity', type=float, default=0.2,
                        help='最小两亲性分数')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备')

    # 调试模式
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')

    return parser.parse_args()

def get_optimal_amp_conditions():
    """
    生成已知有效的抗菌肽条件组合
    
    Returns:
        策略列表，每个策略包含一组条件和温度设置
    """
    # 每个策略包含: 电荷、疏水性、序列长度、温度
    strategies = [
        # 经典抗菌肽 - 高电荷、中等疏水性、中等长度
        {"name": "Classic Cationic AMP", "charge": 4.5, "hydrophobicity": 0.45, "length": 22.0, "temp": 0.8},

        # 短抗菌肽 - 较高电荷、高疏水性、短长度
        {"name": "Short Potent AMP", "charge": 3.0, "hydrophobicity": 0.6, "length": 15.0, "temp": 0.7},

        # 长抗菌肽 - 高电荷、两端两亲性、长序列
        {"name": "Long Transmembrane AMP", "charge": 5.0, "hydrophobicity": 0.4, "length": 28.0, "temp": 0.85},

        # 低毒性AMP - 中等电荷、适中疏水性
        {"name": "Low Toxicity AMP", "charge": 2.5, "hydrophobicity": 0.42, "length": 20.0, "temp": 0.75},

        # 高稳定性AMP - 适中电荷、两亲性
        {"name": "High Stability AMP", "charge": 3.0, "hydrophobicity": 0.48, "length": 24.0, "temp": 0.7},

        # 极短序列AMP - 高电荷密度
        {"name": "Ultra-Short AMP", "charge": 3.0, "hydrophobicity": 0.52, "length": 12.0, "temp": 0.65},

        # 两亲性α螺旋AMP
        {"name": "Amphipathic Alpha-Helix AMP", "charge": 4.0, "hydrophobicity": 0.5, "length": 18.0, "temp": 0.75},

        # 富含色氨酸的AMP - 调整疏水性偏高
        {"name": "Aromatic-Rich AMP", "charge": 2.0, "hydrophobicity": 0.55, "length": 16.0, "temp": 0.7},

        # Beta折叠AMP - 中等电荷，适中疏水性
        {"name": "Beta-Sheet AMP", "charge": 3.5, "hydrophobicity": 0.45, "length": 20.0, "temp": 0.8},

        # 探索性条件 - 高变异度
        {"name": "Exploratory Design", "charge": 4.0, "hydrophobicity": 0.47, "length": 22.0, "temp": 1.0}
    ]

    return strategies

def add_noise_to_strategy(strategy, noise_level=0.1):
    """
    为策略添加随机噪声，增加多样性
    
    Args:
        strategy: 原始策略
        noise_level: 噪声水平
        
    Returns:
        添加噪声后的策略
    """
    noisy_strategy = strategy.copy()

    # 对每个数值参数添加噪声
    noisy_strategy["charge"] += random.uniform(-1.0, 1.0) * noise_level
    noisy_strategy["hydrophobicity"] += random.uniform(-0.1, 0.1) * noise_level
    noisy_strategy["length"] += random.uniform(-3.0, 3.0) * noise_level
    noisy_strategy["temp"] += random.uniform(-0.1, 0.1) * noise_level * 0.5

    # 确保参数在合理范围内
    noisy_strategy["charge"] = max(1.0, min(7.0, noisy_strategy["charge"]))
    noisy_strategy["hydrophobicity"] = max(0.3, min(0.7, noisy_strategy["hydrophobicity"]))
    noisy_strategy["length"] = max(10.0, min(35.0, noisy_strategy["length"]))
    noisy_strategy["temp"] = max(0.5, min(1.2, noisy_strategy["temp"]))

    return noisy_strategy

def generate_adaptive_strategies(num_strategies, prev_results=None):
    """
    根据先前结果自适应生成策略
    
    Args:
        num_strategies: 要生成的策略数量
        prev_results: 先前迭代的结果
        
    Returns:
        策略列表
    """
    base_strategies = get_optimal_amp_conditions()

    if prev_results is None or len(prev_results) == 0:
        # 首次迭代，使用预定义策略并添加噪声
        selected_strategies = random.sample(base_strategies, min(num_strategies, len(base_strategies)))
        return [add_noise_to_strategy(s) for s in selected_strategies]

    # 从之前的结果中学习
    successful_parameters = defaultdict(list)

    # 收集成功序列的参数
    for result in prev_results:
        if result['amp_score'] >= 0.6 and result['toxicity_score'] <= 0.4:
            if 'generation_charge' in result:
                successful_parameters['charge'].append(result['generation_charge'])
                successful_parameters['hydrophobicity'].append(result['generation_hydrophobicity'])
                successful_parameters['length'].append(result['generation_length'])
                successful_parameters['temp'].append(result['generation_temp'])

    strategies = []

    # 如果有足够的成功案例，基于它们创建新策略
    if all(len(v) >= 3 for v in successful_parameters.values()):
        logger.info("Creating new strategies based on previous successful sequences")

        # 计算成功参数的平均值和标准差
        avg_charge = np.mean(successful_parameters['charge'])
        std_charge = np.std(successful_parameters['charge'])

        avg_hydro = np.mean(successful_parameters['hydrophobicity'])
        std_hydro = np.std(successful_parameters['hydrophobicity'])

        avg_length = np.mean(successful_parameters['length'])
        std_length = np.std(successful_parameters['length'])

        avg_temp = np.mean(successful_parameters['temp'])

        # 创建集中在成功区域的策略
        for i in range(num_strategies):
            # 70%的概率使用学习到的参数，30%的概率使用预定义策略以增加多样性
            if random.random() < 0.7:
                charge = random.gauss(avg_charge, std_charge * 1.2)
                hydro = random.gauss(avg_hydro, std_hydro * 1.2)
                length = random.gauss(avg_length, std_length * 1.2)
                temp = avg_temp + random.uniform(-0.1, 0.1)

                # 确保参数在合理范围内
                charge = max(1.0, min(7.0, charge))
                hydro = max(0.3, min(0.7, hydro))
                length = max(10.0, min(35.0, length))
                temp = max(0.5, min(1.2, temp))

                strategy = {
                    "name": f"Adaptive Strategy-{i+1}",
                    "charge": charge,
                    "hydrophobicity": hydro,
                    "length": length,
                    "temp": temp
                }
                strategies.append(strategy)
            else:
                # 使用预定义策略并添加噪声
                base = random.choice(base_strategies)
                strategies.append(add_noise_to_strategy(base, noise_level=0.15))
    else:
        # 没有足够的数据进行学习，回退到预定义策略
        logger.info("Using predefined strategies with random variations")
        selected_strategies = random.sample(base_strategies, min(num_strategies, len(base_strategies)))
        strategies = [add_noise_to_strategy(s, noise_level=0.2) for s in selected_strategies]

    return strategies

def generate_sequences_with_strategy(model, strategy, args, device, batch_idx, iteration, evaluator):
    """
    使用特定策略生成序列
    
    Args:
        model: 扩散模型
        strategy: 生成策略
        args: 命令行参数
        device: 计算设备
        batch_idx: 批次索引
        iteration: 迭代索引
        evaluator: 评估器
        
    Returns:
        生成的序列及其评估结果
    """
    strategy_name = strategy["name"]
    charge = strategy["charge"]
    hydrophobicity = strategy["hydrophobicity"]
    length = strategy["length"]
    temperature = strategy["temp"]

    # 随机确定实际序列长度
    seq_len = int(max(args.seq_len_min, min(args.seq_len_max,
                                           round(length + random.uniform(-2, 2)))))

    logger.info(f"Using strategy: {strategy_name}")
    logger.info(f"Conditions: Charge={charge:.2f}, Hydrophobicity={hydrophobicity:.2f}, " +
                f"Length={length:.1f}, Temperature={temperature:.2f}")
    logger.info(f"Actual sequence length: {seq_len}")

    # 准备条件
    conditions = torch.tensor([
        [charge, hydrophobicity, length]
    ], dtype=torch.float32).to(device)

    current_batch_size = args.batch_size

    # 为当前批次创建条件
    batch_conditions = conditions.repeat(current_batch_size, 1)

    logger.info(f"Generating batch {batch_idx+1}, iteration {iteration+1}, strategy: {strategy_name}")

    try:
        with torch.no_grad():
            generated_batch = model.sample(
                batch_size=current_batch_size,
                seq_len=seq_len,
                condition=batch_conditions,
                device=device,
                temperature=temperature
            )

        # 解码序列
        sequences = []
        for i, tokens in enumerate(tqdm(generated_batch, desc="Decoding sequences")):
            try:
                # 解码
                seq = model.tokenizer.decode(tokens)

                # 如果序列为空，创建一个随机的氨基酸序列作为替代
                if not seq:
                    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
                    seq = ''.join(random.choices(amino_acids, k=seq_len))
                    logger.info(f"Generated random replacement sequence: {seq}")

                sequences.append(seq)
            except Exception as e:
                logger.error(f"Error decoding sequence: {e}")
                # 创建一个随机的氨基酸序列作为替代
                amino_acids = "ACDEFGHIKLMNPQRSTVWY"
                seq = ''.join(random.choices(amino_acids, k=seq_len))
                logger.info(f"Generated random replacement sequence: {seq}")
                sequences.append(seq)

        # 评估序列
        logger.info("Evaluating generated sequences...")
        all_results = []
        for seq in tqdm(sequences, desc="Evaluating sequences"):
            try:
                result = evaluate_amp_properties(seq)
                # 添加生成条件信息用于后续分析
                result['generation_strategy'] = strategy_name
                result['generation_charge'] = charge
                result['generation_hydrophobicity'] = hydrophobicity
                result['generation_length'] = length
                result['generation_temp'] = temperature
                result['iteration'] = iteration + 1
                result['batch'] = batch_idx + 1
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating sequence: {e}")

        return all_results

    except RuntimeError as e:
        logger.error(f"Error generating batch: {e}")
        # 返回空列表
        return []

def main():
    # 获取参数
    args = get_generate_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device)
    logger.info(f"Using device: {args.device}")

    # 加载检查点
    logger.info(f"Loading model from {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    except Exception as e:
        logger.error(f"加载检查点失败: {e}")
        return

    # 获取保存的参数
    saved_args = checkpoint.get('args', {})
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)

    # 初始化分词器
    tokenizer = checkpoint.get('tokenizer', None)
    if tokenizer is None:
        logger.info("Tokenizer not found in checkpoint, initializing a new tokenizer...")
        tokenizer = ProteinTokenizer()

    # 确保条件维度与嵌入维度匹配训练时的设置
    embedding_dim = saved_args.get('embedding_dim', 128)
    condition_dim = saved_args.get('condition_dim', 32)
    hidden_dim = saved_args.get('hidden_dim', 512)
    transformer_layers = saved_args.get('transformer_layers', 6)
    transformer_heads = saved_args.get('transformer_heads', 8)

    # 初始化模型
    model = DiscreteProteinDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        condition_dim=condition_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        num_timesteps=saved_args.get('num_timesteps', 1000),
        beta_schedule=saved_args.get('beta_schedule', 'linear'),
        conditional=True,
        pad_id=tokenizer.pad_id,
        debug=args.debug
    )

    # 设置分词器属性
    model.tokenizer = tokenizer

    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Successfully loaded model weights")
        except Exception as e:
            logger.error(f"加载模型权重出错: {e}")
    else:
        logger.warning("Warning: Model weights not found in checkpoint!")

    model.to(device)
    model.eval()

    # 初始化评估器
    evaluator = AMPEvaluator(tokenizer)

    # 准备存储所有结果
    all_results = []
    all_strategies_results = []

    # 迭代生成和优化
    for iteration in range(args.iterations):
        logger.info(f"\n===== Starting iteration {iteration+1}/{args.iterations} =====")

        # 根据先前结果自适应地生成策略
        adaptive_strategies = generate_adaptive_strategies(
            args.num_strategies,
            prev_results=all_results if iteration > 0 else None
        )

        iteration_results = []

        # 对每个策略生成序列
        samples_per_strategy = args.num_samples // (len(adaptive_strategies) * args.iterations)
        batches_per_strategy = (samples_per_strategy + args.batch_size - 1) // args.batch_size

        for strategy_idx, strategy in enumerate(adaptive_strategies):
            strategy_results = []

            for batch_idx in range(batches_per_strategy):
                batch_results = generate_sequences_with_strategy(
                    model, strategy, args, device, batch_idx, iteration, evaluator
                )

                if batch_results:
                    strategy_results.extend(batch_results)
                    all_results.extend(batch_results)
                    iteration_results.extend(batch_results)

            # 评估当前策略的效果
            if strategy_results:
                success_rate = sum(1 for r in strategy_results
                                 if r['amp_score'] >= args.min_amp_score and
                                    r['toxicity_score'] <= args.max_toxicity) / len(strategy_results)

                logger.info(f"Strategy '{strategy['name']}' success rate: {success_rate:.2f}")

                # 保存策略结果
                strategy_summary = {
                    'strategy_name': strategy['name'],
                    'iteration': iteration + 1,
                    'success_rate': success_rate,
                    'avg_amp_score': np.mean([r['amp_score'] for r in strategy_results]),
                    'avg_toxicity': np.mean([r['toxicity_score'] for r in strategy_results]),
                    'avg_stability': np.mean([r['stability_score'] for r in strategy_results]),
                    'avg_fitness': np.mean([r['fitness'] for r in strategy_results]),
                    'samples_generated': len(strategy_results),
                    'charge': strategy['charge'],
                    'hydrophobicity': strategy['hydrophobicity'],
                    'length': strategy['length'],
                    'temperature': strategy['temp']
                }
                all_strategies_results.append(strategy_summary)

        # 保存当前迭代的结果
        if iteration_results:
            iteration_df = pd.DataFrame(iteration_results)
            iteration_file = os.path.join(
                args.output_dir,
                f"{args.output_prefix}_iteration_{iteration+1}.csv"
            )
            iteration_df.to_csv(iteration_file, index=False)
            logger.info(f"Iteration {iteration+1} results saved to {iteration_file}")

            # 生成报告
            evaluator.generate_report(iteration_results, detailed=False)

    # 完成所有迭代后，保存总结果
    if all_results:
        # 转换为DataFrame
        results_df = pd.DataFrame(all_results)
        strategies_df = pd.DataFrame(all_strategies_results)

        # 保存所有生成的序列
        output_file = os.path.join(args.output_dir, f"{args.output_prefix}_all.csv")
        results_df.to_csv(output_file, index=False)

        # 保存策略效果摘要
        strategies_file = os.path.join(args.output_dir, f"{args.output_prefix}_strategies.csv")
        strategies_df.to_csv(strategies_file, index=False)

        logger.info(f"已保存 {len(results_df)} 个生成的序列到 {output_file}")
        logger.info(f"已保存策略效果摘要到 {strategies_file}")

        # 选择顶级候选物
        try:
            top_candidates = evaluator.select_candidates(
                results_df['sequence'].tolist(),
                top_k=args.top_k,
                min_amp_score=args.min_amp_score,
                max_toxicity=args.max_toxicity,
                min_stability=args.min_stability,
                min_amphipathicity=args.min_amphipathicity
            )

            # 打印顶级候选物
            logger.info("\nTop AMP candidates:")
            for i, candidate in enumerate(top_candidates, 1):
                logger.info(f"{i}. Sequence: {candidate['sequence']}")
                logger.info(f"   Length: {len(candidate['sequence'])}")
                logger.info(f"   AMP Score: {candidate['amp_score']:.3f}")
                logger.info(f"   Toxicity: {candidate['toxicity_score']:.3f}")
                logger.info(f"   Stability: {candidate['stability_score']:.3f}")
                logger.info(f"   Amphipathicity: {candidate['amphipathicity']:.3f}")
                logger.info(f"   Net Charge: {candidate['charge']:.2f}")
                logger.info(f"   Hydrophobicity: {candidate['hydrophobicity']:.3f}")
                logger.info(f"   Fitness: {candidate['fitness']:.3f}")
                logger.info("")

            # 保存顶级候选物
            if top_candidates:
                top_df = pd.DataFrame(top_candidates)
                top_file = os.path.join(args.output_dir, f"{args.output_prefix}_top.csv")
                top_df.to_csv(top_file, index=False)
                logger.info(f"Saved top candidates to {top_file}")

                # 绘制雷达图比较顶级候选物
                radar_file = os.path.join(args.output_dir, f"{args.output_prefix}_radar_plot.png")
                evaluator.generate_radar_plots(
                    top_candidates[:5],  # 展示前5个
                    output_file=radar_file
                )
            else:
                logger.warning("No top candidates found")
        except Exception as e:
            logger.error(f"Error selecting top candidates: {e}")

        # 绘制结果
        try:
            plot_file = os.path.join(args.output_dir, f"{args.output_prefix}_plots.png")
            evaluator.plot_results(
                all_results,
                figsize=(15, 12),
                output_file=plot_file,
                include_correlation=True
            )
            logger.info("Analysis charts saved")

            # 绘制策略效果对比
            if len(all_strategies_results) > 1:
                try:
                    plt.figure(figsize=(12, 8))
                    strategies_df = pd.DataFrame(all_strategies_results)

                    plt.subplot(2, 2, 1)
                    sns.barplot(x='strategy_name', y='success_rate', data=strategies_df)
                    plt.title('Success Rate by Strategy')
                    plt.xlabel('Strategy')
                    plt.ylabel('Success Rate')
                    plt.xticks(rotation=45, ha='right')

                    plt.subplot(2, 2, 2)
                    sns.barplot(x='strategy_name', y='avg_amp_score', data=strategies_df)
                    plt.title('Average Antimicrobial Activity')
                    plt.xlabel('Strategy')
                    plt.ylabel('AMP Score')
                    plt.xticks(rotation=45, ha='right')

                    plt.subplot(2, 2, 3)
                    sns.barplot(x='strategy_name', y='avg_toxicity', data=strategies_df)
                    plt.title('Average Toxicity')
                    plt.xlabel('Strategy')
                    plt.ylabel('Toxicity Score')
                    plt.xticks(rotation=45, ha='right')

                    plt.subplot(2, 2, 4)
                    sns.barplot(x='strategy_name', y='avg_fitness', data=strategies_df)
                    plt.title('Average Fitness')
                    plt.xlabel('Strategy')
                    plt.ylabel('Fitness Score')
                    plt.xticks(rotation=45, ha='right')

                    plt.tight_layout()
                    strategies_plot = os.path.join(args.output_dir, f"{args.output_prefix}_strategies_comparison.png")
                    plt.savefig(strategies_plot, dpi=300, bbox_inches='tight')
                    logger.info(f"Strategy comparison chart saved to {strategies_plot}")
                except Exception as e:
                    logger.error(f"Error drawing charts: {e}")
        except Exception as e:
            logger.error(f"Error drawing charts: {e}")
    else:
        logger.warning("No sequences successfully generated!")


if __name__ == "__main__":
    # 导入可视化库
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        logger.warning("Failed to import visualization library, charts will not be generated")

    main()
