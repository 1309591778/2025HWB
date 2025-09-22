import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert # 【新增】导入希尔伯特变换函数
from scipy.stats import entropy

# ==============================================================================
# 1. 理论故障频率计算函数 (保持不变)
# ==============================================================================
def calculate_theoretical_frequencies(rpm):
    """根据转速计算SKF6205轴承的理论故障频率"""
    if rpm == 0:
        return {'BPFI': 0, 'BPFO': 0, 'BSF': 0}
    # 驱动端轴承为SKF6205，参数见表1
    n_balls = 9
    d_ball = 0.3126
    D_pitch = 1.537
    fr = rpm / 60.0

    bpfi = (n_balls / 2) * fr * (1 + (d_ball / D_pitch))
    bpfo = (n_balls / 2) * fr * (1 - (d_ball / D_pitch))
    bsf = (D_pitch / (2 * d_ball)) * fr * (1 - (d_ball / D_pitch) ** 2)

    return {'BPFI': bpfi, 'BPFO': bpfo, 'BSF': bsf}


# ==============================================================================
# 2. 特征提取核心函数 (核心修改)
# ==============================================================================
def extract_features(segments, labels, rpms, filenames, sample_rate=32000):
    """
    【最终增强版 v2】增加了谐波比、能量熵等高级特征。
    """
    feature_list = []
    print("🚀 开始为所有样本段提取特征 (最终增强版 v2)...")
    n = segments.shape[1]
    freq_axis = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]

    for i, seg in enumerate(segments):
        # (基础信息)
        label, rpm, filename = labels[i], rpms[i], filenames[i]

        # --- 1. 时域特征 (新增 脉冲指标) ---
        rms = np.sqrt(np.mean(seg ** 2))
        # ... (其他时域特征保持不变)
        std_dev = np.std(seg)
        # 【新增】脉冲指标 (Impulse Factor)
        impulse_factor = np.max(np.abs(seg)) / np.mean(np.abs(seg)) if np.mean(np.abs(seg)) != 0 else 0

        # --- 2. 频域特征 (新增 谐波幅值比) ---
        theo_freqs = calculate_theoretical_frequencies(rpm)
        envelope = np.abs(hilbert(seg))
        fft_vals_env = np.abs(np.fft.fft(envelope))[:n // 2]
        freq_features_env = {}
        harmonic_ratios = {}

        # 先计算所有基频和二倍频幅值
        amplitudes = {}
        for f_type, f_val in theo_freqs.items():
            for j in range(1, 3):  # 只需要1倍和2倍频
                target_freq = f_val * j
                idx = np.argmin(np.abs(freq_axis - target_freq))
                amp = fft_vals_env[idx]
                amplitudes[f'{f_type}_{j}x'] = amp
                freq_features_env[f'{f_type}_{j}x_env'] = amp  # 保留原始幅值特征

        # 【新增】计算谐波幅值比
        for f_type in theo_freqs.keys():
            base_amp = amplitudes.get(f'{f_type}_1x', 0)
            harmonic_amp = amplitudes.get(f'{f_type}_2x', 0)
            # 添加 _hr 后缀代表 Harmonic Ratio
            harmonic_ratios[f'{f_type}_hr'] = harmonic_amp / base_amp if base_amp > 1e-6 else 0

        # --- 3. 时频域特征 (新增 能量熵) ---
        wp = pywt.WaveletPacket(data=seg, wavelet='db1', mode='symmetric', maxlevel=3)
        nodes = wp.get_level(3, order='natural')
        wavelet_energy = np.array([np.sum(node.data ** 2) for node in nodes])

        # 【新增】计算小波包能量熵
        # 先对能量进行归一化，使其构成概率分布
        total_energy = np.sum(wavelet_energy)
        energy_dist = wavelet_energy / total_energy if total_energy > 1e-6 else np.zeros_like(wavelet_energy)
        wavelet_entropy = entropy(energy_dist, base=2)

        # --- 4. 整合所有特征 ---
        features = {
            'filename': filename, 'label': label, 'rpm': rpm,
            'rms': rms, 'kurtosis': kurtosis(seg), 'skewness': skew(seg),
            'peak_to_peak': np.max(seg) - np.min(seg),
            'crest_factor': np.max(np.abs(seg)) / rms if rms != 0 else 0,
            'std_dev': std_dev,
            'clearance_factor': np.max(np.abs(seg)) / (np.mean(np.sqrt(np.abs(seg))) ** 2) if np.mean(
                np.sqrt(np.abs(seg))) != 0 else 0,
            'impulse_factor': impulse_factor,
            'wavelet_entropy': wavelet_entropy,
            **freq_features_env,
            **harmonic_ratios,
        }
        # 仍然保留能量分布本身作为特征
        for j, energy in enumerate(wavelet_energy):
            features[f'wavelet_energy_{j}'] = energy

        feature_list.append(features)

    print("✅ 特征提取完成！")
    return pd.DataFrame(feature_list)


# ==============================================================================
# 3. 主程序
# ==============================================================================
if __name__ == "__main__":
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    SEGMENTS_PATH = os.path.join(PROCESSED_DIR, 'source_segments.npy')
    LABELS_PATH = os.path.join(PROCESSED_DIR, 'source_labels.npy')
    RPMS_PATH = os.path.join(PROCESSED_DIR, 'source_rpms.npy')
    # 【新增】定义 filenames 文件的路径
    FILENAMES_PATH = os.path.join(PROCESSED_DIR, 'source_filenames.npy')
    FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features.csv')

    try:
        segments = np.load(SEGMENTS_PATH)
        labels = np.load(LABELS_PATH)
        rpms = np.load(RPMS_PATH)
        # 【新增】加载 filenames 数据
        filenames = np.load(FILENAMES_PATH)

        print(f"成功加载预处理数据: {len(segments)} 个样本段。")

        # 【修改】在调用函数时，传入 filenames
        df_features = extract_features(segments, labels, rpms, filenames)

        print("\n📊 最终特征集预览 (前5行):")
        print(df_features.head())
        print(f"\n特征集维度: {df_features.shape}")
        df_features.to_csv(FEATURES_PATH, index=False)
        print(f"\n💾 最终增强版特征集已保存至: {FEATURES_PATH}")
    except FileNotFoundError as e:
        print(f"‼️ 错误：找不到预处理文件 {e.filename}。请先运行 01 和 02 脚本。")
        exit()