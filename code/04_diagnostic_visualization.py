import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pywt  # 导入小波变换库


# ==============================================================================
# 0. 字体设置函数 (保持不变)
# ==============================================================================
def set_chinese_font():
    """
    强制设置中文字体为 'Microsoft YaHei'，解决中文显示问题。
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已强制设置中文字体为: Microsoft YaHei")


# ==============================================================================
# 1. 理论故障频率计算函数 (保持不变)
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
# 2. 核心可视化函数 (核心修改在图表三部分)
# ==============================================================================
def create_all_diagnostic_plots(segments, labels, rpms, sample_rate=32000):
    """【最终版】分别生成时域、频域、时频域三种分析图表。"""
    print("🚀 开始生成全套诊断可视化图表...")

    # (选取样本部分保持不变)
    try:
        class_representatives = {'N': np.where(labels == 'N')[0][0], 'B': np.where(labels == 'B')[0][0],
                                 'IR': np.where(labels == 'IR')[0][0], 'OR': np.where(labels == 'OR')[0][0]}
        class_titles = {'N': '正常', 'B': '滚动体故障', 'IR': '内圈故障', 'OR': '外圈故障'}
        plot_colors = {'N': 'green', 'B': 'orange', 'IR': 'blue', 'OR': 'red'}
    except IndexError as e:
        print(f"‼️ 错误：数据集中缺少某个类别的样本。 {e}")
        return

    output_dir = os.path.join('..', 'data', 'processed')

    # --- 生成图表一：时域波形对比图 (2x2) (保持不变) ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
    fig1.suptitle('四种轴承状态时域波形对比图', fontsize=22, weight='bold')
    for ax, class_code in zip(axes1.flatten(), ['N', 'B', 'IR', 'OR']):
        idx = class_representatives[class_code]
        segment = segments[idx]
        time_axis = np.arange(len(segment)) / sample_rate
        ax.plot(time_axis, segment, color=plot_colors[class_code], linewidth=1)
        ax.set_title(f'{class_titles[class_code]} ({class_code})', fontsize=16)
        ax.set_xlabel('时间 (s)', fontsize=12)
        ax.set_ylabel('加速度', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path1 = os.path.join(output_dir, '时域波形对比图(四种状态).png')
    plt.savefig(save_path1, dpi=300)
    print(f"\n✅ 图表一已保存至:\n{os.path.abspath(save_path1)}")
    plt.close(fig1)

    # --- 生成图表二：频域频谱对比图 (2x2) (保持不变) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    fig2.suptitle('四种轴承状态频域频谱对比图（颜色编码）', fontsize=22, weight='bold')
    freq_colors = {'BPFO': 'red', 'BPFI': 'green', 'BSF': 'blue'}
    for ax, class_code in zip(axes2.flatten(), ['N', 'B', 'IR', 'OR']):
        idx = class_representatives[class_code]
        segment, rpm = segments[idx], rpms[idx]
        n = len(segment)
        freq_axis = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]
        fft_vals = np.abs(np.fft.fft(segment))[:n // 2]
        ax.plot(freq_axis, fft_vals, color=plot_colors[class_code], linewidth=1)
        ax.set_title(f'{class_titles[class_code]} ({class_code})', fontsize=16)
        ax.set_xlabel('频率 (Hz)', fontsize=12)
        ax.set_ylabel('幅值', fontsize=12)
        ax.set_xlim([0, 600])
        ax.grid(True, linestyle='--', alpha=0.6)
        if class_code != 'N':
            theo_freqs = calculate_theoretical_frequencies(rpm)
            legend_handles = []
            for freq_type, freq_val in theo_freqs.items():
                color = freq_colors[freq_type]
                for i in range(1, 3):  # 绘制1倍频和2倍频
                    freq_line = freq_val * i
                    if freq_line < 600:
                        line = ax.axvline(x=freq_line, color=color, linestyle='--', alpha=0.9)
                        if i == 1: legend_handles.append(line)
            ax.legend(legend_handles, [f'{k}={v:.1f}Hz' for k, v in theo_freqs.items()], loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path2 = os.path.join(output_dir, '频域频谱对比图(颜色编码).png')
    plt.savefig(save_path2, dpi=300)
    print(f"✅ 图表二已保存至:\n{os.path.abspath(save_path2)}")
    plt.close(fig2)

    # --- 【核心修改】生成图表三：时频域分析图 (2x2 小波时频图) ---
    print("🚀 开始生成时频域分析对比图...")
    fig3, axes3 = plt.subplots(2, 2, figsize=(18, 12), sharex=True, sharey=True)
    fig3.suptitle('四种轴承状态小波时频图对比', fontsize=22, weight='bold')

    wavelet = 'cmor1.5-1.0'
    scales = np.arange(1, 512)

    im = None
    for ax, class_code in zip(axes3.flatten(), ['N', 'B', 'IR', 'OR']):
        idx = class_representatives[class_code]
        segment, rpm = segments[idx], rpms[idx]
        time_axis = np.arange(len(segment)) / sample_rate

        coefficients, frequencies = pywt.cwt(segment, scales, wavelet, 1.0 / sample_rate)

        im = ax.pcolormesh(time_axis, frequencies, np.abs(coefficients), cmap='viridis', shading='auto')
        ax.set_title(f'{class_titles[class_code]} ({class_code})', fontsize=16)
        ax.set_ylabel('频率 (Hz)', fontsize=12)
        ax.set_xlabel('时间 (s)', fontsize=12)
        ax.set_ylim([0, 600])

        if class_code != 'N':
            theo_freqs = calculate_theoretical_frequencies(rpm)
            fault_freq_key = {'B': 'BSF', 'IR': 'BPFI', 'OR': 'BPFO'}[class_code]
            fault_freq_val = theo_freqs[fault_freq_key]

            # 【修改】统一使用高对比度的红色，并加粗线条
            line_color = 'red'
            line_width = 2

            # 标记1倍频
            ax.axhline(y=fault_freq_val, color=line_color, linestyle='--', linewidth=line_width,
                       label=f'{fault_freq_key}={fault_freq_val:.1f}Hz')
            # 标记2倍频
            if fault_freq_val * 2 < 600:
                ax.axhline(y=fault_freq_val * 2, color=line_color, linestyle=':', linewidth=line_width,
                           label=f'2x{fault_freq_key}={fault_freq_val * 2:.1f}Hz')
            ax.legend(loc='upper right')

    # 【修改】使用更稳健的方法，手动为颜色条创建位置
    fig3.subplots_adjust(right=0.88)  # 在图的右侧留出空间
    cbar_ax = fig3.add_axes([0.9, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    fig3.colorbar(im, cax=cbar_ax, label='幅值')

    save_path3 = os.path.join(output_dir, '时频域分析对比图(四种状态).png')
    plt.savefig(save_path3, dpi=300)
    print(f"✅ 图表三已保存至:\n{os.path.abspath(save_path3)}")
    plt.close(fig3)


# ==============================================================================
# 3. 主程序 (保持不变)
# ==============================================================================
if __name__ == "__main__":
    set_chinese_font()
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    SEGMENTS_PATH, LABELS_PATH, RPMS_PATH = [os.path.join(PROCESSED_DIR, f) for f in
                                             ['source_segments.npy', 'source_labels.npy', 'source_rpms.npy']]

    try:
        segments, labels, rpms = np.load(SEGMENTS_PATH), np.load(LABELS_PATH), np.load(RPMS_PATH)
        print(f"成功加载预处理数据: {len(segments)} 个样本段。")
        create_all_diagnostic_plots(segments, labels, rpms)
    except FileNotFoundError as e:
        print(f"‼️ 错误：找不到预处理文件 {e.filename}。请先运行 02_data_preprocessing.py。")
        exit()