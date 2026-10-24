import torch
import os
import numpy as np
import pandas as pd
import seaborn as sns
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
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

class AMPEvaluator:
    """抗菌肽评估器"""

    def __init__(self, tokenizer):
        """
        初始化评估器
        
        Args:
            tokenizer: 用于解码序列的分词器
        """
        self.tokenizer = tokenizer

    def evaluate_sequence(self, sequence):
        """
        评估单个序列
        
        Args:
            sequence: 氨基酸序列或token ID序列
            
        Returns:
            评估结果字典
        """
        # 如果输入是token ID序列，先解码
        if isinstance(sequence,
                      torch.Tensor) or (isinstance(sequence, list)
                                        and isinstance(sequence[0], int)):
            sequence = self.tokenizer.decode(sequence)

        # 评估序列属性
        return evaluate_amp_properties(sequence)

    def evaluate_batch(self, sequences):
        """
        评估多个序列
        
        Args:
            sequences: 序列列表
            
        Returns:
            评估结果列表，按适应度排序
        """
        results = []

        for seq in sequences:
            result = self.evaluate_sequence(seq)
            results.append(result)

        # 按适应度分数排序（降序）
        results.sort(key=lambda x: x['fitness'], reverse=True)

        return results

    def select_candidates(self,
                          sequences,
                          top_k=5,
                          min_amp_score=0.6,
                          max_toxicity=0.4,
                          max_weight=5000,
                          min_stability=0.5,
                          min_amphipathicity=0.2):
        """
        筛选候选序列
        
        Args:
            sequences: 序列列表
            top_k: 返回的候选数量
            min_amp_score: 最小抗菌活性分数
            max_toxicity: 最大毒性分数
            max_weight: 最大分子量
            min_stability: 最小稳定性分数
            min_amphipathicity: 最小两亲性分数
            
        Returns:
            筛选后的候选列表
        """
        # 评估所有序列
        results = self.evaluate_batch(sequences)

        # 应用过滤器 - 增加了对稳定性和两亲性的筛选
        filtered = [
            seq for seq in results
            if seq['amp_score'] >= min_amp_score and 
               seq['toxicity_score'] <= max_toxicity and 
               seq['molecular_weight'] <= max_weight and
               seq['stability_score'] >= min_stability and
               seq['amphipathicity'] >= min_amphipathicity
        ]

        # 返回前K个
        return filtered[:top_k]

    def generate_report(self, results, output_file=None, detailed=True):
        """
        生成评估报告
        
        Args:
            results: 评估结果列表
            output_file: 可选的输出文件路径
            detailed: 是否生成详细报告
            
        Returns:
            报告数据框
        """
        # 创建数据框
        df = pd.DataFrame(results)

        # 打印摘要统计
        print("\n=== 抗菌肽评估摘要 ===")
        print(f"评估的序列数量: {len(results)}")
        
        # 基本属性统计
        print("\n--- 基本物理化学属性 ---")
        print(f"序列长度范围: {df['sequence'].str.len().min()}-{df['sequence'].str.len().max()} (平均: {df['sequence'].str.len().mean():.1f})")
        print(f"分子量范围: {df['molecular_weight'].min():.1f}-{df['molecular_weight'].max():.1f} (平均: {df['molecular_weight'].mean():.1f})")
        print(f"净电荷范围: {df['charge'].min():.1f}-{df['charge'].max():.1f} (平均: {df['charge'].mean():.1f})")
        print(f"疏水性范围: {df['hydrophobicity'].min():.3f}-{df['hydrophobicity'].max():.3f} (平均: {df['hydrophobicity'].mean():.3f})")
        print(f"两亲性范围: {df['amphipathicity'].min():.3f}-{df['amphipathicity'].max():.3f} (平均: {df['amphipathicity'].mean():.3f})")
        
        # 稳定性相关统计
        print("\n--- 稳定性参数 ---")
        print(f"不稳定性指数范围: {df['instability_index'].min():.1f}-{df['instability_index'].max():.1f} (平均: {df['instability_index'].mean():.1f})")
        print(f"脂肪族指数范围: {df['aliphatic_index'].min():.1f}-{df['aliphatic_index'].max():.1f} (平均: {df['aliphatic_index'].mean():.1f})")
        print(f"GRAVY值范围: {df['gravy'].min():.3f}-{df['gravy'].max():.3f} (平均: {df['gravy'].mean():.3f})")
        print(f"螺旋倾向性范围: {df['helical_propensity'].min():.3f}-{df['helical_propensity'].max():.3f} (平均: {df['helical_propensity'].mean():.3f})")
        
        # 功能性预测统计
        print("\n--- 功能性预测 ---")
        print(f"抗菌活性分数: {df['amp_score'].mean():.3f} (标准差: {df['amp_score'].std():.3f})")
        print(f"毒性分数: {df['toxicity_score'].mean():.3f} (标准差: {df['toxicity_score'].std():.3f})")
        print(f"稳定性分数: {df['stability_score'].mean():.3f} (标准差: {df['stability_score'].std():.3f})")
        print(f"综合适应度: {df['fitness'].mean():.3f} (标准差: {df['fitness'].std():.3f})")
        
        # 如果要详细报告，打印潜在候选序列信息
        if detailed and len(df) > 0:
            print("\n--- 顶级候选序列 ---")
            top_candidates = df.sort_values('fitness', ascending=False).head(3)
            
            for i, (idx, row) in enumerate(top_candidates.iterrows()):
                print(f"\n候选 #{i+1} - 适应度: {row['fitness']:.3f}")
                print(f"序列: {row['sequence']}")
                print(f"长度: {len(row['sequence'])}, 分子量: {row['molecular_weight']:.1f}, 净电荷: {row['charge']:.1f}")
                print(f"抗菌活性: {row['amp_score']:.3f}, 毒性: {row['toxicity_score']:.3f}, 稳定性: {row['stability_score']:.3f}")
                print(f"两亲性: {row['amphipathicity']:.3f}, 疏水性: {row['hydrophobicity']:.3f}")
                print(f"不稳定性指数: {row['instability_index']:.1f}, 脂肪族指数: {row['aliphatic_index']:.1f}")

        # 如果提供了输出文件，则保存
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\n详细报告已保存到 {output_file}")

        return df

    def plot_results(self, results, figsize=(15, 12), output_file=None, include_correlation=True):
        """
        绘制评估结果图表
        
        Args:
            results: 评估结果列表
            figsize: 图表大小
            output_file: 可选的输出文件路径
            include_correlation: 是否包含相关性热图
        """
        # 创建数据框
        df = pd.DataFrame(results)
        
        if len(df) == 0:
            print("警告: 没有数据可绘制")
            return
            
        # 创建复杂的图表布局
        if include_correlation:
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(3, 3, figure=fig)
            
            ax1 = fig.add_subplot(gs[0, 0])  # AMP Activity vs Toxicity
            ax2 = fig.add_subplot(gs[0, 1])  # Hydrophobicity vs Charge
            ax3 = fig.add_subplot(gs[0, 2])  # Amphipathicity vs Helicity
            ax4 = fig.add_subplot(gs[1, 0])  # Fitness Distribution
            ax5 = fig.add_subplot(gs[1, 1])  # Molecular Weight Distribution
            ax6 = fig.add_subplot(gs[1, 2])  # Stability vs AMP Activity
            ax7 = fig.add_subplot(gs[2, :])  # Correlation Heatmap
        else:
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
            ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]

        # 1. 抗菌活性与毒性的散点图
        ax1.scatter(df['amp_score'], df['toxicity_score'], alpha=0.7, c=df['fitness'], cmap='viridis')
        ax1.set_xlabel('AMP Activity')
        ax1.set_ylabel('Toxicity')
        ax1.set_title('AMP Activity vs Toxicity')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 2. 疏水性与电荷的散点图
        ax2.scatter(df['hydrophobicity'], df['charge'], alpha=0.7, c=df['amp_score'], cmap='plasma')
        ax2.set_xlabel('Hydrophobicity')
        ax2.set_ylabel('Charge')
        ax2.set_title('Hydrophobicity vs Charge')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 3. 两亲性与螺旋倾向性的散点图
        ax3.scatter(df['amphipathicity'], df['helical_propensity'], alpha=0.7, c=df['amp_score'], cmap='plasma')
        ax3.set_xlabel('Amphipathicity')
        ax3.set_ylabel('Helicity')
        ax3.set_title('Amphipathicity vs Helicity')
        ax3.grid(True, linestyle='--', alpha=0.7)

        # 4. 适应度分布直方图
        ax4.hist(df['fitness'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_xlabel('Fitness')
        ax4.set_ylabel('Sequence Count')
        ax4.set_title('Fitness Distribution')
        ax4.grid(True, linestyle='--', alpha=0.7)

        # 5. 分子量分布直方图
        ax5.hist(df['molecular_weight'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
        ax5.set_xlabel('Molecular Weight')
        ax5.set_ylabel('Sequence Count')
        ax5.set_title('Molecular Weight Distribution')
        ax5.grid(True, linestyle='--', alpha=0.7)

        # 6. 稳定性与抗菌活性的散点图
        ax6.scatter(df['stability_score'], df['amp_score'], alpha=0.7, c=df['toxicity_score'], cmap='coolwarm')
        ax6.set_xlabel('Stability')
        ax6.set_ylabel('AMP Activity')
        ax6.set_title('Stability vs AMP Activity')
        ax6.grid(True, linestyle='--', alpha=0.7)
        
        # 7. 相关性热图（如果启用）
        if include_correlation:
            # 选择感兴趣的列进行相关性分析
            corr_cols = ['amp_score', 'toxicity_score', 'stability_score', 'fitness', 
                        'charge', 'hydrophobicity', 'amphipathicity', 'helical_propensity',
                        'instability_index', 'aliphatic_index']
            
            corr = df[corr_cols].corr()
            
            # 绘制热图
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax7)
            ax7.set_title('Attribute Correlation Heatmap')

        plt.tight_layout()

        # 如果提供了输出文件，则保存
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"图表已保存到 {output_file}")

        plt.show()
        
    def generate_radar_plots(self, results, top_n=3, figsize=(15, 5), output_file=None):
        """
        为顶级候选序列生成雷达图比较
        
        Args:
            results: 评估结果列表
            top_n: 要显示的顶级候选数量
            figsize: 图表大小
            output_file: 可选的输出文件路径
        """
        # 创建数据框并排序
        df = pd.DataFrame(results)
        if len(df) == 0:
            print("警告: 没有数据可绘制")
            return
            
        top_candidates = df.sort_values('fitness', ascending=False).head(top_n)
        
        # 设置雷达图
        fig, axes = plt.subplots(1, min(top_n, len(top_candidates)), figsize=figsize, subplot_kw=dict(polar=True))
        
        # 如果只有一个候选，确保axes是可迭代的
        if top_n == 1 or len(top_candidates) == 1:
            axes = [axes]
        
        # 雷达图的类别和数据缩放
        categories = ['AMP Activity', 'Low Toxicity', 'Stability', 'Amphipathicity', 'Helicity', 'Fitness']
        
        for i, (idx, candidate) in enumerate(top_candidates.iterrows()):
            if i >= len(axes):
                break
                
            # 转换数据为雷达图格式
            values = [
                candidate['amp_score'],
                1 - candidate['toxicity_score'],  # 转换为"Low Toxicity"
                candidate['stability_score'],
                min(candidate['amphipathicity'] * 1.2, 1.0),  # 轻微放大两亲性，上限为1
                candidate['helical_propensity'],
                min(candidate['fitness'] * 1.1, 1.0)  # 轻微放大适应度，上限为1
            ]
            
            # 闭合多边形
            values = np.concatenate((values, [values[0]]))
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))
            
            # 绘制雷达图
            ax = axes[i]
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(angles[:-1] * 180/np.pi, categories)
            ax.set_ylim(0, 1)
            
            # 设置标题为序列和适应度
            short_seq = candidate['sequence'][:10] + "..." if len(candidate['sequence']) > 10 else candidate['sequence']
            ax.set_title(f"{short_seq}\nFitness: {candidate['fitness']:.3f}", y=1.2)
            
        plt.tight_layout()
        
        # 如果提供了输出文件，则保存
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"雷达图已保存到 {output_file}")
            
        plt.show()
        
    def interactive_selection(self, results, min_criteria=None):
        """
        提供一种交互式方法来选择候选序列，允许用户调整标准
        
        Args:
            results: 评估结果列表
            min_criteria: 初始最小标准字典，如{'amp_score': 0.6, 'toxicity_score': 0.4}
            
        Returns:
            用户选择的候选序列列表
        """
        # 如果没有提供最小标准，使用默认值
        if min_criteria is None:
            min_criteria = {
                'amp_score': 0.6,
                'toxicity_score': 0.4,  # 这是最大值
                'stability_score': 0.5,
                'amphipathicity': 0.2,
                'fitness': 0.3
            }
            
        df = pd.DataFrame(results)
        if len(df) == 0:
            print("没有可选择的候选序列")
            return []
        
        # 显示数据统计
        print("\n=== 候选序列统计 ===")
        for col in min_criteria.keys():
            if col in df.columns:
                print(f"{col}: 平均 = {df[col].mean():.3f}, 最小 = {df[col].min():.3f}, 最大 = {df[col].max():.3f}")
        
        # 交互式调整标准
        print("\n请为每个标准输入阈值（或按回车保持默认值）:")
        adjusted_criteria = {}
        
        for key, default in min_criteria.items():
            if key == 'toxicity_score':
                prompt = f"{key} (最大值，默认: {default}): "
            else:
                prompt = f"{key} (最小值，默认: {default}): "
                
            try:
                user_input = input(prompt)
                if user_input.strip():
                    adjusted_criteria[key] = float(user_input)
                else:
                    adjusted_criteria[key] = default
            except ValueError:
                print(f"输入有误，将使用默认值 {default}")
                adjusted_criteria[key] = default
        
        # 应用筛选标准
        filtered_df = df.copy()
        
        for key, threshold in adjusted_criteria.items():
            if key == 'toxicity_score':
                # 毒性分数越低越好
                filtered_df = filtered_df[filtered_df[key] <= threshold]
            else:
                # 其他分数越高越好
                filtered_df = filtered_df[filtered_df[key] >= threshold]
        
        # 显示结果
        if len(filtered_df) == 0:
            print("\n没有序列满足所有标准。尝试放宽标准。")
            return []
        
        print(f"\n找到 {len(filtered_df)} 个满足标准的序列。")
        print("\n前5个候选序列:")
        
        top5 = filtered_df.sort_values('fitness', ascending=False).head(5)
        for i, (idx, row) in enumerate(top5.iterrows()):
            print(f"\n候选 #{i+1} - 适应度: {row['fitness']:.3f}")
            print(f"序列: {row['sequence']}")
            print(f"抗菌活性: {row['amp_score']:.3f}, 毒性: {row['toxicity_score']:.3f}, 稳定性: {row['stability_score']:.3f}")
            print(f"两亲性: {row['amphipathicity']:.3f}, 螺旋倾向性: {row['helical_propensity']:.3f}")
        
        return filtered_df.sort_values('fitness', ascending=False).to_dict('records')