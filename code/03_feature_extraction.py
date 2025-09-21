import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert # 【新增】导入希尔伯特变换函数

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
def extract_features(segments, labels, rpms, sample_rate=32000):
    """
    为每个信号段提取多维度特征。
    【最终版：同时包含原始谱和包络谱的频域特征】
    """
    feature_list = []
    print("🚀 开始为所有样本段提取特征 (最终增强版)...")

    n = segments.shape[1]
    freq_axis = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]

    for i, seg in enumerate(segments):
        label = labels[i]
        rpm = rpms[i]

        # --- 1. 时域特征 (保持不变) ---
        rms = np.sqrt(np.mean(seg ** 2))
        kurt = kurtosis(seg)
        sk = skew(seg)
        peak_to_peak = np.max(seg) - np.min(seg)
        crest_factor = np.max(np.abs(seg)) / rms if rms != 0 else 0
        std_dev = np.std(seg)
        clearance_factor = np.max(np.abs(seg)) / (np.mean(np.sqrt(np.abs(seg))) ** 2) if np.mean(
            np.sqrt(np.abs(seg))) != 0 else 0

        # --- 2. 频域特征 (核心修改：分为两部分) ---
        theo_freqs = calculate_theoretical_frequencies(rpm)

        # === 2.1 基于原始信号的频域特征 (保留) ===
        fft_vals_raw = np.abs(np.fft.fft(seg))[:n // 2]
        freq_features_raw = {}
        for f_type, f_val in theo_freqs.items():
            for j in range(1, 4):
                target_freq = f_val * j
                idx = np.argmin(np.abs(freq_axis - target_freq))
                # 添加 _raw 后缀以区分
                freq_features_raw[f'{f_type}_{j}x_raw'] = fft_vals_raw[idx]

        # === 2.2 【新增】基于包络谱的频域特征 ===
        envelope = np.abs(hilbert(seg))
        fft_vals_env = np.abs(np.fft.fft(envelope))[:n // 2]
        freq_features_env = {}
        for f_type, f_val in theo_freqs.items():
            for j in range(1, 4):
                target_freq = f_val * j
                idx = np.argmin(np.abs(freq_axis - target_freq))
                # 添加 _env 后缀以区分
                freq_features_env[f'{f_type}_{j}x_env'] = fft_vals_env[idx]

        # --- 3. 时频域特征 (保持不变) ---
        wp = pywt.WaveletPacket(data=seg, wavelet='db1', mode='symmetric', maxlevel=3)
        nodes = wp.get_level(3, order='natural')
        wavelet_energy = [np.sum(node.data ** 2) for node in nodes]

        # --- 4. 整合所有特征 ---
        features = {
            'label': label, 'rpm': rpm,
            # 时域特征
            'rms': rms, 'kurtosis': kurt, 'skewness': sk,
            'peak_to_peak': peak_to_peak, 'crest_factor': crest_factor,
            'std_dev': std_dev, 'clearance_factor': clearance_factor,
            # 两种频域特征
            **freq_features_raw,
            **freq_features_env,
        }
        # 时频域特征
        for j, energy in enumerate(wavelet_energy):
            features[f'wavelet_energy_{j}'] = energy

        feature_list.append(features)

        if (i + 1) % 1000 == 0:
            print(f"  - 已处理 {i + 1}/{len(segments)} 个样本...")

    print("✅ 特征提取完成！")
    return pd.DataFrame(feature_list)

# ==============================================================================
# 3. 主程序 (保持不变)
# ==============================================================================
if __name__ == "__main__":
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    SEGMENTS_PATH = os.path.join(PROCESSED_DIR, 'source_segments.npy')
    LABELS_PATH = os.path.join(PROCESSED_DIR, 'source_labels.npy')
    RPMS_PATH = os.path.join(PROCESSED_DIR, 'source_rpms.npy')
    FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features.csv')

    try:
        segments, labels, rpms = np.load(SEGMENTS_PATH), np.load(LABELS_PATH), np.load(RPMS_PATH)
        print(f"成功加载预处理数据: {len(segments)} 个样本段。")
        df_features = extract_features(segments, labels, rpms)
        print("\n📊 最终特征集预览 (前5行):")
        print(df_features.head())
        print(f"\n特征集维度: {df_features.shape}")
        df_features.to_csv(FEATURES_PATH, index=False)
        print(f"\n💾 最终增强版特征集已保存至: {FEATURES_PATH}")
    except FileNotFoundError as e:
        print(f"‼️ 错误：找不到预处理文件 {e.filename}。请先运行 01 和 02 脚本。")
        exit()