# 03_code.py
import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert
from scipy.stats import entropy
from scipy.signal import find_peaks


# ==============================================================================
# 1. 理论故障频率计算函数 (MODIFIED: 改为返回归一化频率)
# ==============================================================================
def calculate_normalized_frequencies():
    """
    返回SKF6205轴承的归一化故障特征频率（相对于转频 fr 的倍数，无量纲）
    """
    n_balls = 9
    d_ball = 0.3126
    D_pitch = 1.537
    bpfi_norm = (n_balls / 2) * (1 + (d_ball / D_pitch))
    bpfo_norm = (n_balls / 2) * (1 - (d_ball / D_pitch))
    bsf_norm = (D_pitch / (2 * d_ball)) * (1 - (d_ball / D_pitch) ** 2)
    return {'BPFI_norm': bpfi_norm, 'BPFO_norm': bpfo_norm, 'BSF_norm': bsf_norm}


# ==============================================================================
# 2. 安全计算函数 (简化版)
# ==============================================================================
def safe_divide(a, b, default=0):
    """安全除法运算"""
    try:
        return float(a / b) if abs(b) > 1e-8 else default
    except:
        return default


# ==============================================================================
# 3. 特征提取核心函数 (MODIFIED: 使用归一化频率)
# ==============================================================================
def extract_features(segments, labels, rpms, filenames, sample_rate=32000):
    """
    稳定版特征提取函数 + 专业缺失值处理准备
    """
    feature_list = []
    print("🚀 开始为所有样本段提取特征...")
    n = segments.shape[1]
    freq_axis = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]

    # === MODIFIED ===
    # 预先计算归一化频率（无量纲，与RPM无关）
    norm_freqs = calculate_normalized_frequencies()
    # === MODIFIED END ===

    for i, seg in enumerate(segments):
        # 基础信息
        label, rpm, filename = labels[i], rpms[i], filenames[i]

        # === MODIFIED ===
        # 计算转频 (Hz)
        if rpm == 0:
            fr = 1.0  # 避免除零，但后续特征会为0
        else:
            fr = rpm / 60.0
        # === MODIFIED END ===

        # 时域特征
        rms = np.sqrt(np.mean(seg ** 2))
        std_dev = np.std(seg)
        impulse_factor = np.max(np.abs(seg)) / np.mean(np.abs(seg)) if np.mean(np.abs(seg)) != 0 else 0

        # 频域特征
        envelope = np.abs(hilbert(seg))
        fft_vals_env = np.abs(np.fft.fft(envelope))[:n // 2]
        freq_features_env = {}
        harmonic_ratios = {}
        amplitudes = {}

        # === MODIFIED ===
        # 使用归一化频率 * 转频 = 绝对频率
        for f_type_norm, f_norm in norm_freqs.items():
            f_type = f_type_norm.replace('_norm', '')  # 'BPFI_norm' -> 'BPFI'
            for j in range(1, 3):  # 1倍和2倍频
                target_freq = (f_norm * j) * fr  # 归一化频率 * 转频 = 绝对频率
                if 0 <= target_freq < sample_rate / 2 and fr > 0:  # 边界检查
                    idx = np.argmin(np.abs(freq_axis - target_freq))
                    amp = fft_vals_env[idx] if idx < len(fft_vals_env) else 0
                    amplitudes[f'{f_type}_{j}x'] = amp
                    freq_features_env[f'{f_type}_{j}x_env'] = amp
        # === MODIFIED END ===

        # 计算谐波幅值比
        for f_type in ['BPFI', 'BPFO', 'BSF']:
            base_amp = amplitudes.get(f'{f_type}_1x', 0)
            harmonic_amp = amplitudes.get(f'{f_type}_2x', 0)
            harmonic_ratios[f'{f_type}_hr'] = safe_divide(harmonic_amp, base_amp)

        # 时频域特征
        try:
            wp = pywt.WaveletPacket(data=seg, wavelet='db1', mode='symmetric', maxlevel=3)
            nodes = wp.get_level(3, order='natural')
            wavelet_energy = np.array([np.sum(node.data ** 2) for node in nodes])
            # 小波包能量熵
            total_energy = np.sum(wavelet_energy)
            if total_energy > 1e-6:
                energy_dist = wavelet_energy / total_energy
                energy_dist = np.clip(energy_dist, 1e-10, 1)  # 避免log(0)
                wavelet_entropy = entropy(energy_dist, base=2)
            else:
                wavelet_entropy = 0
        except Exception as e:
            # 如果小波包分解失败，使用默认值
            wavelet_energy = np.zeros(8)
            wavelet_entropy = 0

        # ======================================================================
        # 新增：N类样本增强特征
        # ======================================================================
        n_class_specific_features = {}
        if label == 'N':
            try:
                # 平稳性特征：自相关衰减率
                autocorr = np.correlate(seg, seg, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]
                # 计算自相关衰减率（前100个点）
                decay_window = min(100, len(autocorr))
                if decay_window > 1:
                    decay_rate = np.sum(np.abs(np.diff(autocorr[:decay_window]))) / decay_window
                else:
                    decay_rate = 0
                # 噪声水平特征
                diff_seg = np.diff(seg)
                if len(diff_seg) > 0:
                    noise_level = safe_divide(np.std(diff_seg), np.std(seg))
                else:
                    noise_level = 0
                # 冲击指标（正常信号应该很少有冲击）
                envelope = np.abs(hilbert(seg))
                impulse_indicator = safe_divide(np.max(envelope), np.mean(envelope))
                n_class_specific_features = {
                    'N_autocorr_decay': float(decay_rate) if not np.isnan(decay_rate) else 0,
                    'N_noise_level': float(noise_level) if not np.isnan(noise_level) else 0,
                    'N_impulse_indicator': float(impulse_indicator) if not np.isnan(impulse_indicator) else 0
                }
            except:
                # 如果计算失败，使用默认值
                n_class_specific_features = {
                    'N_autocorr_decay': 0,
                    'N_noise_level': 0,
                    'N_impulse_indicator': 0
                }
        # ======================================================================

        # 回归：只保留经过验证的稳定特征
        # ======================================================================
        # 整合所有特征（确保所有值都是有限数值）
        features = {
            'filename': filename, 'label': label, 'rpm': rpm,
            # 基础统计特征
            'rms': float(rms) if np.isfinite(rms) else 0,
            'kurtosis': float(kurtosis(seg)) if np.isfinite(kurtosis(seg)) else 0,
            'skewness': float(skew(seg)) if np.isfinite(skew(seg)) else 0,
            'peak_to_peak': float(np.max(seg) - np.min(seg)) if np.isfinite(np.max(seg) - np.min(seg)) else 0,
            'crest_factor': float(np.max(np.abs(seg)) / rms) if rms != 0 and np.isfinite(rms) else 0,
            'std_dev': float(std_dev) if np.isfinite(std_dev) else 0,
            'clearance_factor': float(np.max(np.abs(seg)) / (np.mean(np.sqrt(np.abs(seg))) ** 2)) if np.mean(
                np.sqrt(np.abs(seg))) != 0 else 0,
            'impulse_factor': float(impulse_factor) if np.isfinite(impulse_factor) else 0,
            'wavelet_entropy': float(wavelet_entropy) if np.isfinite(wavelet_entropy) else 0,
            **{k: float(v) if np.isfinite(v) else 0 for k, v in freq_features_env.items()},
            **{k: float(v) if np.isfinite(v) else 0 for k, v in harmonic_ratios.items()},
            # N类专属特征（新增）
            **{k: float(v) if np.isfinite(v) else 0 for k, v in n_class_specific_features.items()},
        }
        # 小波能量特征
        for j, energy in enumerate(wavelet_energy):
            features[f'wavelet_energy_{j}'] = float(energy) if np.isfinite(energy) else 0
        feature_list.append(features)
        # 进度显示
        if (i + 1) % 5000 == 0:
            print(f"  - 已处理 {i + 1}/{len(segments)} 个样本段")
    print("✅ 特征提取完成！")
    return pd.DataFrame(feature_list)


# ==============================================================================
# 4. 主程序 (增强版：添加专业缺失值处理)
# ==============================================================================
if __name__ == "__main__":
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    SEGMENTS_PATH = os.path.join(PROCESSED_DIR, 'source_segments.npy')
    LABELS_PATH = os.path.join(PROCESSED_DIR, 'source_labels.npy')
    RPMS_PATH = os.path.join(PROCESSED_DIR, 'source_rpms.npy')
    FILENAMES_PATH = os.path.join(PROCESSED_DIR, 'source_filenames.npy')
    FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features.csv')
    try:
        segments = np.load(SEGMENTS_PATH)
        labels = np.load(LABELS_PATH)
        rpms = np.load(RPMS_PATH)
        filenames = np.load(FILENAMES_PATH)
        print(f"成功加载预处理数据: {len(segments)} 个样本段。")
        df_features = extract_features(segments, labels, rpms, filenames)
        print("\n📊 特征集预览 (前5行):")
        print(df_features.head())
        print(f"\n特征集维度: {df_features.shape}")
        # 检查NaN值
        nan_count = df_features.isnull().sum().sum()
        inf_count = np.isinf(df_features.select_dtypes(include=[np.number])).sum().sum()
        print(f"  - NaN值统计: {nan_count} 个")
        print(f"  - 无穷值统计: {inf_count} 个")
        # 使用专业方法处理缺失值和无穷值
        if nan_count > 0 or inf_count > 0:
            print("  - 发现异常值，正在进行专业处理...")
            # 将无穷值替换为NaN
            df_features = df_features.replace([np.inf, -np.inf], np.nan)
            # 使用均值填充NaN值
            numeric_columns = df_features.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df_features[col].isnull().any():
                    mean_value = df_features[col].mean()
                    df_features[col].fillna(mean_value, inplace=True)
            print("  - 异常值处理完成")
        df_features.to_csv(FEATURES_PATH, index=False)
        print(f"\n💾 特征集已保存至: {FEATURES_PATH}")
    except FileNotFoundError as e:
        print(f"‼️ 错误：找不到预处理文件 {e.filename}。请先运行 01 和 02 脚本。")
        exit()