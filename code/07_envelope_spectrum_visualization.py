import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.signal import hilbert


# ==============================================================================
# 0. 字体设置函数 (复用)
# ==============================================================================
def set_chinese_font():
    """
    强制设置中文字体为 'Microsoft YaHei'，解决中文显示问题。
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已强制设置中文字体为: Microsoft YaHei")


# ==============================================================================
# 1. 理论故障频率计算函数 (复用)
# ==============================================================================
def calculate_theoretical_frequencies(rpm):
    """根据转速计算SKF6205轴承的理论故障频率"""
    n_balls, d_ball, D_pitch = 9, 0.3126, 1.537
    fr = rpm / 60.0
    bpfo = (n_balls / 2) * fr * (1 - (d_ball / D_pitch))
    bpfi = (n_balls / 2) * fr * (1 + (d_ball / D_pitch))
    bsf = (D_pitch / (2 * d_ball)) * fr * (1 - (d_ball / D_pitch) ** 2)
    return {'BPFI': bpfi, 'BPFO': bpfo, 'BSF': bsf}


# ==============================================================================
# 2. 核心可视化函数
# ==============================================================================
def create_envelope_comparison_plot(segments, labels, rpms, sample_rate=32000):
    """
    生成原始频谱 vs. 包络频谱的对比图，以展示包络分析的有效性。
    """
    print("🚀 开始生成包络谱分析对比图...")

    # 1. 选取一个滚动体故障(B)样本进行分析
    try:
        fault_idx = np.where(labels == 'B')[0][0]
    except IndexError:
        print("‼️ 错误：数据集中找不到'B'类别的样本，无法生成对比图。")
        return

    segment = segments[fault_idx]
    rpm = rpms[fault_idx]

    # 2. 创建1x2的子图用于对比
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('滚动体故障(B) - 原始频谱 vs. 包络频谱对比', fontsize=22, weight='bold')

    # --- 左图：原始信号频谱 ---
    n = len(segment)
    freq_axis = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]
    fft_vals_raw = np.abs(np.fft.fft(segment))[:n // 2]

    axes[0].plot(freq_axis, fft_vals_raw, color='orange')
    axes[0].set_title('原始信号频谱', fontsize=16)
    axes[0].set_xlabel('频率 (Hz)', fontsize=12)
    axes[0].set_ylabel('幅值', fontsize=12)
    axes[0].set_xlim([0, 600])
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # --- 右图：包络信号频谱 ---
    envelope = np.abs(hilbert(segment))
    fft_vals_env = np.abs(np.fft.fft(envelope))[:n // 2]

    axes[1].plot(freq_axis, fft_vals_env, color='purple')
    axes[1].set_title('包络信号频谱 (特征更清晰)', fontsize=16)
    axes[1].set_xlabel('频率 (Hz)', fontsize=12)
    axes[1].set_xlim([0, 600])
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. 在两张图上都标记理论频率
    theo_freqs = calculate_theoretical_frequencies(rpm)
    bsf_val = theo_freqs['BSF']

    for ax in axes:
        # 只标记BSF及其谐波
        line1 = ax.axvline(x=bsf_val, color='blue', linestyle='--', label=f'BSF={bsf_val:.1f}Hz')
        line2 = ax.axvline(x=bsf_val * 2, color='blue', linestyle=':', label=f'2xBSF={bsf_val * 2:.1f}Hz')
        ax.legend(handles=[line1, line2])

    # 4. 保存图像
    output_dir = os.path.join('..', 'data', 'processed')
    save_path = os.path.join(output_dir, '包络谱分析对比图(滚动体故障).png')
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ 包络谱对比图已成功生成并保存至:\n{os.path.abspath(save_path)}")
    plt.show()


# ==============================================================================
# 3. 主程序
# ==============================================================================
if __name__ == "__main__":
    set_chinese_font()
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    SEGMENTS_PATH, LABELS_PATH, RPMS_PATH = [os.path.join(PROCESSED_DIR, f) for f in
                                             ['source_segments.npy', 'source_labels.npy', 'source_rpms.npy']]

    try:
        segments, labels, rpms = np.load(SEGMENTS_PATH), np.load(LABELS_PATH), np.load(RPMS_PATH)
        print(f"成功加载预处理数据: {len(segments)} 个样本段。")
        create_envelope_comparison_plot(segments, labels, rpms)
    except FileNotFoundError as e:
        print(f"‼️ 错误：找不到预处理文件 {e.filename}。请先运行 02_data_preprocessing.py。")
        exit()