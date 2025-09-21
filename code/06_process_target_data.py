import os
import numpy as np
import pandas as pd
import scipy.io
import pywt
from scipy.stats import kurtosis, skew


# ==============================================================================
# 1. 工具函数 (从之前的脚本复用)
# ==============================================================================
def load_mat_file(file_path: str):
    """读取 .mat 文件"""
    try:
        return scipy.io.loadmat(file_path)
    except NotImplementedError:
        import h5py
        mat_data = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                mat_data[key] = np.array(f[key])
        return mat_data
    except Exception as e:
        print(f"读取文件 {os.path.basename(file_path)} 时发生错误: {e}")
        return None


def calculate_theoretical_frequencies(rpm):
    """根据转速计算SKF6205轴承的理论故障频率"""
    # 注意：目标域轴承型号未知，此处沿用源域轴承参数作为近似
    # 这是迁移学习中的一个常见假设：基础物理特性相似
    if rpm == 0:
        return {'BPFI': 0, 'BPFO': 0, 'BSF': 0}
    n_balls, d_ball, D_pitch = 9, 0.3126, 1.537
    fr = rpm / 60.0
    bpfi = (n_balls / 2) * fr * (1 + (d_ball / D_pitch))
    bpfo = (n_balls / 2) * fr * (1 - (d_ball / D_pitch))
    bsf = (D_pitch / (2 * d_ball)) * fr * (1 - (d_ball / D_pitch) ** 2)
    return {'BPFI': bpfi, 'BPFO': bpfo, 'BSF': bsf}


# ==============================================================================
# 2. 目标域数据处理与特征提取核心函数
# ==============================================================================
def process_target_data(target_dir, segment_len=4096, stride=512):
    """
    加载、分段并提取所有目标域文件的特征。
    """
    all_features_list = []

    # 根据赛题描述，设定目标域的固定参数
    target_sr = 32000  # 采样频率为32kHz
    target_rpm = 600  # 轴承转速约600 rpm

    print("🚀 开始处理目标域数据...")

    # 准备FFT的频率轴
    n = segment_len
    freq_axis = np.fft.fftfreq(n, 1 / target_sr)[:n // 2]

    # 扫描目标文件夹
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.mat')])

    for filename in target_files:
        file_path = os.path.join(target_dir, filename)

        # 1. 加载文件并提取信号
        mat_data = load_mat_file(file_path)
        if not mat_data:
            continue

        # 目标域文件内部变量名未知，我们假设信号是其中最长的数值数组
        signal_key = max((k for k in mat_data if isinstance(mat_data[k], np.ndarray) and mat_data[k].ndim > 1),
                         key=lambda k: mat_data[k].shape[0], default=None)

        if not signal_key:
            print(f"  - 警告：在文件 {filename} 中未找到可用的信号数据，已跳过。")
            continue
        signal = mat_data[signal_key].flatten()

        # 2. 信号分段 (使用与源域相同的参数)
        num_segments = 0
        for i in range(0, len(signal) - segment_len + 1, stride):
            segment = signal[i: i + segment_len]
            num_segments += 1

            # --- 3. 特征提取 (与03脚本的逻辑完全一致) ---
            # 时域特征
            rms = np.sqrt(np.mean(segment ** 2))
            kurt = kurtosis(segment)
            sk = skew(segment)
            peak_to_peak = np.max(segment) - np.min(segment)
            crest_factor = np.max(np.abs(segment)) / rms if rms != 0 else 0

            # 频域特征
            fft_vals = np.abs(np.fft.fft(segment))[:n // 2]
            theo_freqs = calculate_theoretical_frequencies(target_rpm)
            freq_features = {}
            for f_type, f_val in theo_freqs.items():
                for j in range(1, 4):
                    target_freq = f_val * j
                    idx = np.argmin(np.abs(freq_axis - target_freq))
                    freq_features[f'{f_type}_{j}x'] = fft_vals[idx]

            # 时频域特征
            wp = pywt.WaveletPacket(data=segment, wavelet='db1', mode='symmetric', maxlevel=3)
            nodes = wp.get_level(3, order='natural')
            wavelet_energy = [np.sum(node.data ** 2) for node in nodes]

            # 整合所有特征
            features = {
                'source_file': filename,  # 记录样本来源文件
                'rpm': target_rpm,
                'rms': rms, 'kurtosis': kurt, 'skewness': sk,
                'peak_to_peak': peak_to_peak, 'crest_factor': crest_factor,
                **freq_features,
            }
            for j, energy in enumerate(wavelet_energy):
                features[f'wavelet_energy_{j}'] = energy

            all_features_list.append(features)

        print(f"  - ✅ 已处理: {filename} -> 生成了 {num_segments} 个样本段。")

    print("\n✅ 目标域数据特征提取完成！")
    return pd.DataFrame(all_features_list)


# ==============================================================================
# 3. 主程序
# ==============================================================================
if __name__ == "__main__":
    # --- 输入路径 ---
    TARGET_DATA_DIR = os.path.join('..', 'data', 'target')

    # --- 输出路径 ---
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    TARGET_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'target_features.csv')

    # 1. 执行处理与特征提取
    df_target_features = process_target_data(TARGET_DATA_DIR)

    if not df_target_features.empty:
        print("\n📊 目标域特征集预览 (前5行):")
        print(df_target_features.head())
        print(f"\n特征集维度: {df_target_features.shape}")

        # 2. 保存特征集
        df_target_features.to_csv(TARGET_FEATURES_PATH, index=False)
        print(f"\n💾 目标域特征集已保存至: {TARGET_FEATURES_PATH}")
        print("\n🎉 任务一的全部工作已圆满完成！")
    else:
        print("\n未能在目标目录中处理任何文件。")