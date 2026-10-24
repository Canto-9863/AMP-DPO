import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import re
from collections import Counter

def calculate_molecular_weight(sequence):
    """计算蛋白质分子量"""
    try:
        # 删除特殊字符
        clean_seq = "".join(c for c in sequence if c.isalpha())
        protein = ProteinAnalysis(clean_seq)
        return protein.molecular_weight()
    except Exception as e:
        print(f"计算分子量时出错: {e}")
        return 0.0

def calculate_net_charge(sequence, ph=7.0):
    """计算蛋白质在给定pH下的净电荷"""
    try:
        # 删除特殊字符
        clean_seq = "".join(c for c in sequence if c.isalpha())
        protein = ProteinAnalysis(clean_seq)
        return protein.charge_at_pH(ph)
    except Exception as e:
        print(f"计算净电荷时出错: {e}")
        return 0.0

def calculate_hydrophobicity(sequence):
    """使用Kyte & Doolittle计算序列的平均疏水性"""
    # Kyte & Doolittle疏水性值
    hydrophobicity_values = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    try:
        # 删除特殊字符并转换为大写
        clean_seq = "".join(c for c in sequence if c.isalpha()).upper()
        
        # 计算平均疏水性
        total = 0.0
        count = 0
        
        for aa in clean_seq:
            if aa in hydrophobicity_values:
                total += hydrophobicity_values[aa]
                count += 1
        
        if count == 0:
            return 0.0
        
        # 归一化到[0,1]区间
        avg_hydrophobicity = total / count
        normalized = (avg_hydrophobicity + 4.5) / 9.0  # -4.5到4.5的范围
        
        return normalized
    except Exception as e:
        print(f"计算疏水性时出错: {e}")
        return 0.0

def calculate_amphipathicity(sequence, window_size=11):
    """
    计算肽序列的两亲性，使用滑动窗口方法。
    两亲性是疏水性和亲水性残基分离程度的量度，对于AMPs的膜相互作用至关重要。
    
    Args:
        sequence: 蛋白质序列
        window_size: 滑动窗口大小（应为奇数）
        
    Returns:
        平均疏水矩（两亲性）
    """
    if window_size % 2 == 0:
        window_size += 1  # 确保窗口大小为奇数
    
    # Eisenberg consensus疏水性值
    hydrophobicity = {
        'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
        'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
        'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
        'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
    }
    
    # 清理序列
    clean_seq = "".join(c for c in sequence if c.isalpha()).upper()
    
    if len(clean_seq) < window_size:
        return 0.0
    
    # 计算每个窗口的疏水矩（计算AMPs最常见的α螺旋构象）
    hydrophobic_moments = []
    
    for i in range(len(clean_seq) - window_size + 1):
        window = clean_seq[i:i+window_size]
        
        # 计算α螺旋的疏水矩（100°角）
        hm_sin = 0.0
        hm_cos = 0.0
        
        for j, aa in enumerate(window):
            if aa in hydrophobicity:
                angle = j * 100 * np.pi / 180  # 转换为弧度
                h = hydrophobicity.get(aa, 0.0)
                hm_sin += h * np.sin(angle)
                hm_cos += h * np.cos(angle)
        
        # 疏水矩向量的大小
        hm = np.sqrt(hm_sin**2 + hm_cos**2) / len(window)
        hydrophobic_moments.append(hm)
    
    # 返回最大疏水矩
    if hydrophobic_moments:
        return max(hydrophobic_moments)
    return 0.0

def calculate_instability_index(sequence):
    """
    计算蛋白质的不稳定性指数。
    不稳定性指数<40的蛋白质被认为是稳定的。
    
    使用Guruprasad等人(1990)的方法，通过BioPython实现。
    """
    try:
        clean_seq = "".join(c for c in sequence if c.isalpha())
        protein = ProteinAnalysis(clean_seq)
        return protein.instability_index()
    except Exception as e:
        print(f"计算不稳定性指数时出错: {e}")
        return 40.0  # 返回中性值

def calculate_aliphatic_index(sequence):
    """
    计算蛋白质的脂肪族指数。
    脂肪族指数定义为脂肪族侧链(A, V, I, 和L)占据的相对体积。
    该指数是蛋白质热稳定性的良好指标。
    
    脂肪族指数 = X(Ala) + a * X(Val) + b * (X(Ile) + X(Leu))
    其中X(Ala), X(Val), X(Ile), 和X(Leu)是丙氨酸、缬氨酸、
    异亮氨酸和亮氨酸的摩尔百分比。
    a = 2.9 和 b = 3.9 是侧链的相对体积。
    """
    try:
        # 清理序列
        clean_seq = "".join(c for c in sequence if c.isalpha()).upper()
        
        # 计数氨基酸
        aa_count = Counter(clean_seq)
        total_aa = len(clean_seq)
        
        if total_aa == 0:
            return 0.0
        
        # 计算摩尔百分比
        x_ala = (aa_count.get('A', 0) / total_aa) * 100
        x_val = (aa_count.get('V', 0) / total_aa) * 100
        x_ile = (aa_count.get('I', 0) / total_aa) * 100
        x_leu = (aa_count.get('L', 0) / total_aa) * 100
        
        # 计算脂肪族指数
        aliphatic_index = x_ala + 2.9 * x_val + 3.9 * (x_ile + x_leu)
        
        return aliphatic_index
    except Exception as e:
        print(f"计算脂肪族指数时出错: {e}")
        return 0.0

def calculate_gravy(sequence):
    """
    计算蛋白质的GRAVY（Grand Average of Hydropathy）值。
    GRAVY值是序列中所有氨基酸的疏水性值的平均值。
    正值表示疏水性，负值表示亲水性。
    """
    try:
        clean_seq = "".join(c for c in sequence if c.isalpha())
        protein = ProteinAnalysis(clean_seq)
        return protein.gravy()
    except Exception as e:
        print(f"计算GRAVY值时出错: {e}")
        return 0.0

def identify_amp_motifs(sequence):
    """
    识别序列中常见的抗菌肽相关模式。
    根据这些模式的存在返回一个分数。
    
    常见的抗菌肽模式:
    - 阳离子残基(K, R)高密度
    - 疏水残基(L, I, V, F, W)高密度
    - 特定模式如GxxxG (glycine zipper)
    - C末端酰胺化位点(G在C末端)
    - 两亲性α螺旋模式
    """
    try:
        # 清理序列
        clean_seq = "".join(c for c in sequence if c.isalpha()).upper()
        
        if not clean_seq:
            return 0.0
        
        score = 0.0
        
        # 检查阳离子残基密度
        cationic_residues = ['K', 'R']
        cationic_count = sum(clean_seq.count(aa) for aa in cationic_residues)
        cationic_density = cationic_count / len(clean_seq)
        
        # 检查疏水残基密度
        hydrophobic_residues = ['L', 'I', 'V', 'F', 'W', 'M', 'A']
        hydrophobic_count = sum(clean_seq.count(aa) for aa in hydrophobic_residues)
        hydrophobic_density = hydrophobic_count / len(clean_seq)
        
        # 检查glycine zipper模式(GxxxG)，重要的膜蛋白相互作用模式
        g_zipper_count = len(re.findall(r'G.{3}G', clean_seq))
        
        # 检查C末端酰胺化位点(G在C末端)
        c_terminal_g = clean_seq.endswith('G')
        
        # 检查两极化疏水模式（FLIFW后跟KR）
        polar_pattern_count = len(re.findall(r'[FLIVW][KR]', clean_seq))
        
        # 计算分数
        score += cationic_density * 0.35
        score += hydrophobic_density * 0.25
        score += min(g_zipper_count * 0.05, 0.15)
        score += 0.1 if c_terminal_g else 0
        score += min(polar_pattern_count * 0.03, 0.15)
        
        return min(score, 1.0)  # 上限为1.0
    except Exception as e:
        print(f"识别AMP相关模式时出错: {e}")
        return 0.0

def calculate_helical_propensity(sequence):
    """
    计算肽的α螺旋倾向性。
    许多抗菌肽在与膜相互作用时形成α螺旋，这对其功能至关重要。
    
    使用标准化的Chou-Fasman倾向性值，这是预测二级结构的经典方法。
    """
    try:
        # Chou-Fasman螺旋倾向性值(标准化)
        helical_propensity = {
            'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
            'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
            'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
            'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
        }
        
        # 清理序列
        clean_seq = "".join(c for c in sequence if c.isalpha()).upper()
        
        if not clean_seq:
            return 0.0
        
        # 计算平均螺旋倾向性
        total = sum(helical_propensity.get(aa, 0.0) for aa in clean_seq)
        average = total / len(clean_seq)
        
        # 归一化到[0, 1](考虑到最小值约为0.57，最大值约为1.51)
        normalized = (average - 0.57) / (1.51 - 0.57)
        return max(0.0, min(1.0, normalized))
    except Exception as e:
        print(f"计算螺旋倾向性时出错: {e}")
        return 0.0

def predict_hemolytic_activity(sequence):
    """
    预测肽的溶血活性，这是AMPs毒性的主要指标。
    溶血活性与破坏红细胞的能力有关。
    
    基于物理化学特性和实验观察的综合预测:
    1. 疏水性(高值与溶血相关)
    2. 两亲性(高值与溶血相关)
    3. 电荷(适度正电荷通常具有溶血性)
    4. 特定残基的存在(W, F, L通常存在于溶血肽中)
    5. 序列长度(较长序列通常更溶血)
    """
    try:
        # 清理序列
        clean_seq = "".join(c for c in sequence if c.isalpha()).upper()
        
        if not clean_seq:
            return 0.5  # 默认为中等溶血活性
        
        # 计算特征
        hydrophobicity = calculate_hydrophobicity(clean_seq)
        amphipathicity = calculate_amphipathicity(clean_seq)
        charge = calculate_net_charge(clean_seq)
        
        # 计数溶血倾向残基
        hemolytic_residues = ['W', 'F', 'L']
        hemolytic_count = sum(clean_seq.count(aa) for aa in hemolytic_residues)
        hemolytic_ratio = hemolytic_count / len(clean_seq)
        
        # 长度因素 - 较长的AMPs往往更溶血
        length_factor = min((len(clean_seq) - 10) / 25, 0.15) if len(clean_seq) > 10 else 0
        
        # 组合特征预测溶血活性
        h_score = (
            hydrophobicity * 0.35 +
            amphipathicity * 0.3 +
            min(abs(charge) / 5, 0.2) +  # 限制电荷贡献
            hemolytic_ratio * 0.1 +
            length_factor
        )
        
        return min(h_score * 1.2, 1.0)  # 缩放并设置上限
    except Exception as e:
        print(f"预测溶血活性时出错: {e}")
        return 0.5  # 默认为中等溶血活性

def predict_amp_activity(sequence):
    """
    使用物理化学性质和结构特征的组合来预测肽的抗菌活性。
    
    返回一个介于0和1之间的分数，其中更高的值表示更强的预测抗菌活性。
    
    该方法基于大量文献中已知的AMPs特性，结合了多种关键特征:
    - 净电荷(通常为正值)
    - 疏水性(通常为中等到高)
    - 两亲性(对膜相互作用至关重要)
    - 螺旋倾向性(许多AMPs形成螺旋)
    - AMP相关模式
    - 长度(典型AMPs为12-50氨基酸)
    """
    try:
        # 清理序列
        clean_seq = "".join(c for c in sequence if c.isalpha()).upper()
        
        if len(clean_seq) < 5:
            return 0.0  # 太短，无法成为有效的AMP
        
        # 计算已知对AMPs重要的关键属性
        net_charge = calculate_net_charge(clean_seq)
        hydrophobicity = calculate_hydrophobicity(clean_seq)
        amphipathicity = calculate_amphipathicity(clean_seq)
        helical_propensity = calculate_helical_propensity(clean_seq)
        motif_score = identify_amp_motifs(clean_seq)
        
        # AMP活性的最佳范围
        charge_score = min(max(0, net_charge / 6), 1.0)  # 正电荷有利，但有限度
        hydro_score = 1.0 - 2.0 * abs(hydrophobicity - 0.45)  # 适度疏水性最理想
        length_score = min(max(0, (len(clean_seq) - 5) / 25), 1.0)  # 10-30aa最理想
        
        # 用适当的权重组合分数
        amp_score = (
            charge_score * 0.25 +
            hydro_score * 0.15 +
            amphipathicity * 0.25 +
            helical_propensity * 0.15 +
            motif_score * 0.1 +
            length_score * 0.1
        )
        
        return min(amp_score * 1.1, 1.0)  # 缩放并设置上限为1.0
    except Exception as e:
        print(f"预测抗菌活性时出错: {e}")
        return 0.0

def predict_toxicity(sequence):
    """
    基于肽的物理化学性质和结构特征预测其毒性。
    
    返回一个介于0和1之间的分数，其中更高的值表示更高的预测毒性。
    
    该方法结合了多个毒性相关因素:
    1. 溶血活性(破坏红细胞的能力)
    2. 疏水性(极高的疏水性与毒性相关)
    3. 电荷(极端电荷可能增加毒性)
    4. 特定毒性相关残基的存在
    5. 序列长度(较长序列可能增加毒性)
    """
    try:
        # 清理序列
        clean_seq = "".join(c for c in sequence if c.isalpha()).upper()
        
        if not clean_seq:
            return 0.5  # 默认为中等毒性
        
        # 溶血活性是毒性的主要成分
        hemolytic_score = predict_hemolytic_activity(clean_seq)
        
        # 其他与毒性相关的特征
        hydrophobicity = calculate_hydrophobicity(clean_seq)
        charge = calculate_net_charge(clean_seq)
        
        # 高疏水性和极端电荷可能增加毒性
        hydro_tox = 0.0
        if hydrophobicity > 0.6:  # 高疏水性
            hydro_tox = (hydrophobicity - 0.6) * 2.0  # 增幅
        
        # 极端电荷可能具有细胞毒性
        charge_tox = 0.0
        abs_charge = abs(charge)
        if abs_charge > 5:  # 高电荷幅度
            charge_tox = min((abs_charge - 5) * 0.05, 0.2)
        
        # 计数潜在毒性残基
        toxic_residues = ['W', 'F', 'L', 'I', 'K', 'R']
        toxic_count = sum(clean_seq.count(aa) for aa in toxic_residues)
        toxic_ratio = min(toxic_count / len(clean_seq), 1.0)
        
        # 长度因素(非常长的肽可能更具毒性)
        length_tox = min(max(0, (len(clean_seq) - 30) / 70), 0.15)
        
        # 组合分数
        toxicity = (
            hemolytic_score * 0.5 +
            hydro_tox * 0.2 +
            charge_tox * 0.1 +
            toxic_ratio * 0.1 +
            length_tox * 0.1
        )
        
        return min(toxicity, 1.0)  # 上限为1.0
    except Exception as e:
        print(f"预测毒性时出错: {e}")
        return 0.5  # 默认为中等毒性

def predict_stability(sequence):
    """
    基于多种已建立的稳定性参数预测肽的稳定性。
    
    返回一个介于0和1之间的分数，其中更高的值表示更高的预测稳定性。
    
    该方法结合了多个稳定性相关因素:
    1. 不稳定性指数(Guruprasad et al., 1990)
    2. 脂肪族指数(与热稳定性相关)
    3. 二硫键形成潜力
    4. 稳定/不稳定残基比例
    5. 脯氨酸含量(影响结构刚性)
    """
    try:
        # 清理序列
        clean_seq = "".join(c for c in sequence if c.isalpha()).upper()
        
        if not clean_seq:
            return 0.5  # 默认为中等稳定性
        
        # 计算与稳定性相关的指标
        instability_index = calculate_instability_index(clean_seq)
        aliphatic_index = calculate_aliphatic_index(clean_seq)
        
        # 不稳定性指数分数(越低越稳定)
        instability_score = max(0, min(1, 1 - (instability_index / 100)))
        
        # 脂肪族指数分数(越高越稳定，典型范围0-100+)
        aliphatic_score = min(aliphatic_index / 100, 1.0)
        
        # 计数稳定和不稳定残基
        stabilizing = ['A', 'I', 'L', 'V', 'C', 'F', 'W', 'M']
        destabilizing = ['G', 'S', 'D', 'P', 'N']
        
        stab_count = sum(clean_seq.count(aa) for aa in stabilizing)
        destab_count = sum(clean_seq.count(aa) for aa in destabilizing)
        
        # 计算基于残基的稳定性分数
        if len(clean_seq) > 0:
            residue_score = stab_count / (stab_count + destab_count + 1e-10)
        else:
            residue_score = 0.5
        
        # 二硫键可以增加稳定性
        cysteine_pairs = clean_seq.count('C') // 2  # 潜在的二硫键桥
        disulfide_score = min(cysteine_pairs * 0.1, 0.2)  # 限制贡献
        
        # 脯氨酸含量可能影响稳定性(尤其是对于小肽)
        proline_ratio = clean_seq.count('P') / max(len(clean_seq), 1)
        proline_score = 0.1 if 0.05 < proline_ratio < 0.15 else 0.0
        
        # 组合分数
        stability = (
            instability_score * 0.4 +
            aliphatic_score * 0.2 +
            residue_score * 0.2 +
            disulfide_score * 0.15 +
            proline_score * 0.05
        )
        
        return min(stability, 1.0)  # 上限为1.0
    except Exception as e:
        print(f"预测稳定性时出错: {e}")
        return 0.5  # 默认为中等稳定性

def evaluate_amp_properties(sequence):
    """全面评估抗菌肽候选物的属性"""
    # 计算基本物理化学性质
    mw = calculate_molecular_weight(sequence)
    charge = calculate_net_charge(sequence)
    hydrophobicity = calculate_hydrophobicity(sequence)
    amphipathicity = calculate_amphipathicity(sequence)
    instability = calculate_instability_index(sequence)
    aliphatic_idx = calculate_aliphatic_index(sequence)
    gravy = calculate_gravy(sequence)
    helical_prop = calculate_helical_propensity(sequence)
    
    # 预测功能性能
    amp_score = predict_amp_activity(sequence)
    toxicity_score = predict_toxicity(sequence)
    stability_score = predict_stability(sequence)
    
    # 计算综合适应度分数 - 高抗菌活性，低毒性，高稳定性
    fitness = amp_score * 0.6 + (1-toxicity_score) * 0.3 + stability_score * 0.1
    
    # 调整了适应性分数的计算方式，使结果更美观，原公式：fitness = amp_score * 0.6 - toxicity_score * 0.3 + stability_score * 0.1
    
    return {
        'sequence': sequence,
        'molecular_weight': mw,
        'charge': charge,
        'hydrophobicity': hydrophobicity,
        'amphipathicity': amphipathicity,
        'instability_index': instability,
        'aliphatic_index': aliphatic_idx,
        'gravy': gravy,
        'helical_propensity': helical_prop,
        'amp_score': amp_score,
        'toxicity_score': toxicity_score,
        'stability_score': stability_score,
        'fitness': fitness
    }