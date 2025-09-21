import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import kurtosis, skew


# ==============================================================================
# 1. 理论故障频率计算 (与可视化脚本一致)
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
# 2. 特征提取核心函数
# ==============================================================================
def extract_features(segments, labels, rpms, sample_rate=32000):
    """
    为每个信号段提取多维度特征。
    """
    feature_list = []
    print("🚀 开始为所有样本段提取特征...")

    # 为FFT计算准备频率轴
    n = segments.shape[1]
    freq_axis = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]

    for i, seg in enumerate(segments):
        label = labels[i]
        rpm = rpms[i]

        # --- 1. 时域特征 ---
        rms = np.sqrt(np.mean(seg ** 2))
        kurt = kurtosis(seg)
        sk = skew(seg)
        peak_to_peak = np.max(seg) - np.min(seg)
        crest_factor = np.max(np.abs(seg)) / rms if rms != 0 else 0

        # --- 2. 频域特征 ---
        fft_vals = np.abs(np.fft.fft(seg))[:n // 2]

        # 计算理论频率
        theo_freqs = calculate_theoretical_frequencies(rpm)

        # 提取理论频率及其谐波(2x, 3x)的幅值
        freq_features = {}
        for f_type, f_val in theo_freqs.items():
            if f_val == 0:
                for j in range(1, 4):
                    freq_features[f'{f_type}_{j}x'] = 0
                continue

            for j in range(1, 4):  # 1, 2, 3倍频
                target_freq = f_val * j
                # 找到最接近理论频率的FFT频点的幅值
                idx = np.argmin(np.abs(freq_axis - target_freq))
                freq_features[f'{f_type}_{j}x'] = fft_vals[idx]

        # --- 3. 时频域特征 (小波包变换能量) ---
        wp = pywt.WaveletPacket(data=seg, wavelet='db1', mode='symmetric', maxlevel=3)
        nodes = wp.get_level(3, order='natural')
        wavelet_energy = [np.sum(node.data ** 2) for node in nodes]

        # --- 整合所有特征 ---
        features = {
            'label': label,
            'rpm': rpm,
            'rms': rms,
            'kurtosis': kurt,
            'skewness': sk,
            'peak_to_peak': peak_to_peak,
            'crest_factor': crest_factor,
            **freq_features,  # 合并频域特征字典
        }
        # 合并小波能量特征
        for j, energy in enumerate(wavelet_energy):
            features[f'wavelet_energy_{j}'] = energy

        feature_list.append(features)

        if (i + 1) % 1000 == 0:
            print(f"  - 已处理 {i + 1}/{len(segments)} 个样本...")

    print("✅ 特征提取完成！")
    return pd.DataFrame(feature_list)


# ==============================================================================
# 3. 主程序
# ==============================================================================
if __name__ == "__main__":
    # --- 输入路径 ---
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    SEGMENTS_PATH = os.path.join(PROCESSED_DIR, 'source_segments.npy')
    LABELS_PATH = os.path.join(PROCESSED_DIR, 'source_labels.npy')
    RPMS_PATH = os.path.join(PROCESSED_DIR, 'source_rpms.npy')

    # --- 输出路径 ---
    FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features.csv')

    # 1. 加载预处理好的数据
    try:
        segments = np.load(SEGMENTS_PATH)
        labels = np.load(LABELS_PATH)
        rpms = np.load(RPMS_PATH)
        print(f"成功加载预处理数据: {len(segments)} 个样本段。")
    except FileNotFoundError as e:
        print(f"‼️ 错误：找不到预处理文件 {e.filename}。请先运行 02_data_preprocessing.py。")
        exit()

    # 2. 执行特征提取
    df_features = extract_features(segments, labels, rpms)

    print("\n📊 特征集预览 (前5行):")
    print(df_features.head())
    print(f"\n特征集维度: {df_features.shape}")

    # 3. 保存特征集
    df_features.to_csv(FEATURES_PATH, index=False)
    print(f"\n💾 最终特征集已保存至: {FEATURES_PATH}")
    print("\n🎉 任务一：数据分析与特征提取的全部工作已完成！")