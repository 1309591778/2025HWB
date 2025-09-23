import os
import numpy as np
import pandas as pd
import scipy.io
import pywt
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import hilbert


# ==============================================================================
# 1. 工具函数 (与03脚本完全一致)
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
    """根据转速计算理论故障频率，使用源域轴承参数作为近似"""
    if rpm == 0:
        return {'BPFI': 0, 'BPFO': 0, 'BSF': 0}
    n_balls, d_ball, D_pitch = 9, 0.3126, 1.537
    fr = rpm / 60.0
    bpfi = (n_balls / 2) * fr * (1 + (d_ball / D_pitch))
    bpfo = (n_balls / 2) * fr * (1 - (d_ball / D_pitch))
    bsf = (D_pitch / (2 * d_ball)) * fr * (1 - (d_ball / D_pitch) ** 2)
    return {'BPFI': bpfi, 'BPFO': bpfo, 'BSF': bsf}


# ==============================================================================
# 2. 安全计算函数 (新增)
# ==============================================================================
def safe_divide(a, b, default=0):
    """安全除法运算"""
    try:
        return float(a / b) if abs(b) > 1e-8 else default
    except:
        return default


def safe_sqrt(x, default=0):
    """安全平方根运算"""
    try:
        return float(np.sqrt(abs(x))) if abs(x) > 1e-8 else default
    except:
        return default


# ==============================================================================
# 3. 目标域数据处理与特征提取核心函数 (增强版)
# ==============================================================================
def process_target_data(target_dir, segment_len=4096, stride=512):
    """
    加载、分段并提取所有目标域文件的特征，确保与源域特征提取逻辑完全一致。
    """
    all_features_list = []

    # [cite_start]根据赛题描述，设定目标域的固定参数 [cite: 41]
    target_sr = 32000
    target_rpm = 600

    print("🚀 开始处理目标域数据...")
    n = segment_len
    freq_axis = np.fft.fftfreq(n, 1 / target_sr)[:n // 2]
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.mat')])

    for filename in target_files:
        file_path = os.path.join(target_dir, filename)
        mat_data = load_mat_file(file_path)
        if not mat_data:
            continue

        signal_key = max((k for k in mat_data if isinstance(mat_data[k], np.ndarray) and mat_data[k].ndim > 1),
                         key=lambda k: mat_data[k].shape[0], default=None)
        if not signal_key:
            print(f"  - 警告：在文件 {filename} 中未找到可用的信号数据，已跳过。")
            continue
        signal = mat_data[signal_key].flatten()

        num_segments = 0
        for i in range(0, len(signal) - segment_len + 1, stride):
            segment = signal[i: i + segment_len]
            num_segments += 1

            # --- 特征提取 (与03脚本的最终增强版逻辑完全一致) ---
            # 1. 时域特征
            rms = np.sqrt(np.mean(segment ** 2))
            kurt = kurtosis(segment)
            sk = skew(segment)
            peak_to_peak = np.max(segment) - np.min(segment)
            crest_factor = np.max(np.abs(segment)) / rms if rms != 0 else 0
            std_dev = np.std(segment)
            clearance_factor = np.max(np.abs(segment)) / (np.mean(np.sqrt(np.abs(segment))) ** 2) if np.mean(
                np.sqrt(np.abs(segment))) != 0 else 0
            impulse_factor = np.max(np.abs(segment)) / np.mean(np.abs(segment)) if np.mean(np.abs(segment)) != 0 else 0

            # 2. 频域特征 (基于包络谱，并新增谐波幅值比)
            theo_freqs = calculate_theoretical_frequencies(target_rpm)
            envelope = np.abs(hilbert(segment))
            fft_vals_env = np.abs(np.fft.fft(envelope))[:n // 2]
            freq_features_env = {}
            harmonic_ratios = {}
            amplitudes = {}
            for f_type, f_val in theo_freqs.items():
                for j in range(1, 3):
                    target_freq = f_val * j
                    idx = np.argmin(np.abs(freq_axis - target_freq))
                    amp = fft_vals_env[idx]
                    amplitudes[f'{f_type}_{j}x'] = amp
                    freq_features_env[f'{f_type}_{j}x_env'] = amp
            for f_type in theo_freqs.keys():
                base_amp = amplitudes.get(f'{f_type}_1x', 0)
                harmonic_amp = amplitudes.get(f'{f_type}_2x', 0)
                harmonic_ratios[f'{f_type}_hr'] = harmonic_amp / base_amp if base_amp > 1e-6 else 0

            # 3. 时频域特征 (新增能量熵)
            wp = pywt.WaveletPacket(data=segment, wavelet='db1', mode='symmetric', maxlevel=3)
            nodes = wp.get_level(3, order='natural')
            wavelet_energy = np.array([np.sum(node.data ** 2) for node in nodes])
            total_energy = np.sum(wavelet_energy)
            energy_dist = wavelet_energy / total_energy if total_energy > 1e-6 else np.zeros_like(wavelet_energy)
            wavelet_entropy = entropy(energy_dist, base=2)

            # ======================================================================
            # 新增：基础增强特征 (与03脚本保持一致)
            # ======================================================================

            # 1. 改进的时域特征
            margin_factor = safe_divide(np.max(np.abs(segment)), np.mean(np.abs(segment)) ** 2)
            shape_factor = safe_divide(rms, np.mean(np.abs(segment)))

            # 2. 改进的频域特征
            spectral_features = {}
            if len(fft_vals_env) > 0:
                # 频谱重心
                spectral_centroid = safe_divide(np.sum(freq_axis * fft_vals_env), np.sum(fft_vals_env))
                # 频谱方差
                spectral_spread = safe_divide(np.sum(((freq_axis - spectral_centroid) ** 2) * fft_vals_env),
                                              np.sum(fft_vals_env))
                # 频谱偏度
                spectral_skewness = safe_divide(np.sum(((freq_axis - spectral_centroid) ** 3) * fft_vals_env),
                                                np.sum(fft_vals_env) * (safe_sqrt(spectral_spread) ** 3))
                # 频谱峰度
                spectral_kurtosis = safe_divide(np.sum(((freq_axis - spectral_centroid) ** 4) * fft_vals_env),
                                                np.sum(fft_vals_env) * (spectral_spread ** 2))

                spectral_features = {
                    'spectral_centroid': spectral_centroid,
                    'spectral_spread': spectral_spread,
                    'spectral_skewness': spectral_skewness,
                    'spectral_kurtosis': spectral_kurtosis
                }

            # 3. 包络分析特征
            envelope_features = {}
            if len(envelope) > 0:
                envelope_mean = np.mean(envelope)
                envelope_std = np.std(envelope)
                envelope_rms = np.sqrt(np.mean(envelope ** 2))

                envelope_features = {
                    'envelope_mean': envelope_mean,
                    'envelope_std': envelope_std,
                    'envelope_rms': envelope_rms,
                    'envelope_crest_factor': safe_divide(np.max(envelope), envelope_rms),
                    'envelope_impulse_factor': safe_divide(np.max(envelope), envelope_mean)
                }

            # ======================================================================
            # 新增：N类样本增强特征 (与03脚本保持一致)
            # ======================================================================
            n_class_specific_features = {}
            try:
                # 平稳性特征：自相关衰减率
                autocorr = np.correlate(segment, segment, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]
                # 计算自相关衰减率（前100个点）
                decay_window = min(100, len(autocorr))
                if decay_window > 1:
                    decay_rate = np.sum(np.abs(np.diff(autocorr[:decay_window]))) / decay_window
                else:
                    decay_rate = 0

                # 噪声水平特征
                diff_seg = np.diff(segment)
                if len(diff_seg) > 0:
                    noise_level = safe_divide(np.std(diff_seg), np.std(segment))
                else:
                    noise_level = 0

                # 冲击指标（正常信号应该很少有冲击）
                envelope = np.abs(hilbert(segment))
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

            # 整合所有特征 (原有代码)
            features_full = {
                'source_file': filename, 'rpm': target_rpm,
                # 原有特征
                'rms': rms, 'kurtosis': kurt, 'skewness': sk,
                'peak_to_peak': peak_to_peak, 'crest_factor': crest_factor,
                'std_dev': std_dev, 'clearance_factor': clearance_factor,
                'impulse_factor': impulse_factor,
                'wavelet_entropy': wavelet_entropy,
                **freq_features_env,
                **harmonic_ratios,
                # 新增基础特征
                'margin_factor': margin_factor,
                'shape_factor': shape_factor,
                **spectral_features,
                **envelope_features,
                # N类专属特征（新增）
                **n_class_specific_features,
            }
            for j, energy in enumerate(wavelet_energy):
                features_full[f'wavelet_energy_{j}'] = energy

            # --- 【新增】只保留任务一选定的特征 ---
            # 确保 selected_feature_names 在函数作用域内可用，或者作为参数传入
            # 这里假设 selected_feature_names 是全局加载的，或者通过其他方式传递
            # 为简化，我们在这里直接筛选，但更健壮的做法是在函数参数中传递 selected_feature_names
            # 或者在函数内部重新加载（不推荐，效率低）
            # 临时解决方案：假设 selected_feature_names 已在主程序开始时加载为全局变量
            global selected_feature_names  # 声明使用全局变量
            features_selected = {'source_file': filename, 'rpm': target_rpm}  # 保留必要信息
            for fname in selected_feature_names:
                if fname in features_full:
                    features_selected[fname] = features_full[fname]
                else:
                    print(f"  - 警告：在目标域特征中未找到任务一选定的特征 '{fname}'，将填充为0。")
                    features_selected[fname] = 0  # 或者 np.nan, 根据模型容忍度决定
            # --- 【新增结束】 ---

            # all_features_list.append(features) # 注释掉旧的添加方式
            all_features_list.append(features_selected)  # 添加筛选后的特征

        print(f"  - ✅ 已处理: {filename} -> 生成了 {num_segments} 个样本段。")

    print("\n✅ 目标域数据特征提取完成！")
    return pd.DataFrame(all_features_list)


# ==============================================================================
# 3. 主程序
# ==============================================================================
if __name__ == "__main__":
    TARGET_DATA_DIR = os.path.join('..', 'data', 'target')
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    TARGET_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'target_features.csv')

    # --- 【新增】加载任务一筛选出的特征名称 ---
    SELECTED_FEATURES_PATH = os.path.join('..', 'data', 'processed', 'selected_feature_names.txt')
    try:
        with open(SELECTED_FEATURES_PATH, 'r') as f:
            # 读取特征名并去除换行符
            selected_feature_names = [line.strip() for line in f.readlines()]
        print(f"✅ 成功加载任务一筛选出的特征列表，共 {len(selected_feature_names)} 个特征。")
        # print(f"特征列表: {selected_feature_names}") # 可选：打印确认
    except FileNotFoundError:
        print(f"‼️ 错误：找不到任务一的特征列表文件 {SELECTED_FEATURES_PATH}。请先运行任务一的特征筛选脚本。")
        exit(1)
    except Exception as e:
        print(f"‼️ 加载特征列表时发生错误: {e}")
        exit(1)
    # --- 【新增结束】 ---

    df_target_features = process_target_data(TARGET_DATA_DIR)

    if not df_target_features.empty:
        print("\n📊 目标域特征集预览 (前5行):")
        print(df_target_features.head())
        print(f"\n特征集维度: {df_target_features.shape}")
        df_target_features.to_csv(TARGET_FEATURES_PATH, index=False)
        print(f"\n💾 目标域全特征集已保存至: {TARGET_FEATURES_PATH}")
    else:
        print("\n未能在目标目录中处理任何文件。")