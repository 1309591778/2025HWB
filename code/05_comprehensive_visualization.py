# 10_task1_final_visualizations.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import scipy.io
import scipy.signal
import pywt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ==============================================================================
# 0. 辅助函数
# ==============================================================================
def set_chinese_font():
    """直接从项目文件夹加载指定的字体文件。"""
    font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'SourceHanSansSC-Regular.otf')
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")


def calculate_theoretical_frequencies(rpm):
    """根据转速计算SKF6205轴承的理论故障频率"""
    n_balls, d_ball, D_pitch = 9, 0.3126, 1.537
    fr = rpm / 60.0
    bpfo = (n_balls / 2) * fr * (1 - (d_ball / D_pitch))
    bpfi = (n_balls / 2) * fr * (1 + (d_ball / D_pitch))
    bsf = (D_pitch / (2 * d_ball)) * fr * (1 - (d_ball / D_pitch) ** 2)
    return {'BPFI': bpfi, 'BPFO': bpfo, 'BSF': bsf}


# ==============================================================================
# 1. 新增：时频域分析可视化函数 (小波包变换 WPT)
# ==============================================================================
def visualize_wpt_energy_packets(segments, labels, rpms, sample_rate, output_dir):
    """
    对信号进行小波包变换(WPT)并可视化其能量分布。绘制在一张图上。
    """
    print(f"  - 正在生成小波包能量图...")

    class_representatives = {k: np.where(labels == k)[0][0] for k in ['N', 'B', 'IR', 'OR']}
    class_titles = {'N': '正常', 'B': '滚动体故障', 'IR': '内圈故障', 'OR': '外圈故障'}
    plot_colors = {'N': 'green', 'B': 'orange', 'IR': 'blue', 'OR': 'red'}
    class_codes = ['N', 'B', 'IR', 'OR']

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle('四种轴承状态小波包变换 (WPT) 能量分布对比图', fontsize=22, weight='bold')

    for ax, class_code in zip(axes.flatten(), class_codes):
        idx = class_representatives[class_code]
        segment = segments[idx]
        class_title = class_titles[class_code]
        plot_color = plot_colors[class_code]

        # 1. 执行小波包变换 (3层分解)
        wp = pywt.WaveletPacket(data=segment, wavelet='db1', mode='symmetric', maxlevel=3)

        # 2. 获取第3层的所有节点
        nodes = wp.get_level(3, order='natural')

        # 3. 计算每个节点的能量 (节点数据的平方和)
        packet_energies = np.array([np.sum(node.data ** 2) for node in nodes])

        # 4. 创建时间轴 (与原始信号长度一致)
        time_axis = np.arange(len(segment)) / sample_rate

        # 5. 绘图
        # --- 绘制原始信号 ---
        ax.plot(time_axis, segment, color=plot_color, alpha=0.7, linewidth=0.8, label='原始信号')

        # --- 绘制能量包络 ---
        num_packets = len(packet_energies)
        if num_packets > 0:
            segment_length_per_packet = len(segment) // num_packets
            energy_envelope = np.repeat(packet_energies, segment_length_per_packet)
            if len(energy_envelope) < len(segment):
                energy_envelope = np.pad(energy_envelope, (0, len(segment) - len(energy_envelope)), constant_values=0)
            elif len(energy_envelope) > len(segment):
                energy_envelope = energy_envelope[:len(segment)]

            ax.fill_between(time_axis, 0, -energy_envelope / np.max(energy_envelope) * np.max(np.abs(segment)) * 0.3,
                            color=plot_color, alpha=0.4, label='WPT能量包络 (示意)')

        ax.set_title(f'{class_title} ({class_code}) - 小波包变换 (WPT) 能量分布', fontsize=16, weight='bold')
        ax.set_xlabel('时间 (s)', fontsize=12)
        ax.set_ylabel('加速度 / 能量 (示意)', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_wpt = os.path.join(output_dir, f'图5-WPT能量分布对比.png')
    plt.savefig(save_path_wpt, dpi=300)
    plt.close(fig)
    print(f"  - ✅ 小波包能量图已保存至: {save_path_wpt}")


# ==============================================================================
# 1. 新增：小波时频图（CWT）可视化函数 —— 【已修正】
# ==============================================================================
def visualize_cwt_time_frequency(segments, labels, rpms, sample_rate, output_dir):
    """
    使用连续小波变换（CWT）生成时频图，并标注理论故障频率。绘制在一张图上。
    """
    print(f"  - 正在生成小波时频图...")

    class_titles = {'N': '正常', 'B': '滚动体故障', 'IR': '内圈故障', 'OR': '外圈故障'}
    plot_colors = {'N': 'green', 'B': 'orange', 'IR': 'blue', 'OR': 'red'}
    class_codes = ['N', 'B', 'IR', 'OR']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('四种轴承状态小波时频图对比', fontsize=22, weight='bold')

    # 初始化全局最大最小值
    global_min = float('inf')
    global_max = float('-inf')

    for ax, class_code in zip(axes.flatten(), class_codes):
        idx = np.where(labels == class_code)[0][0]
        segment = segments[idx]
        class_title = class_titles[class_code]

        # 时间轴
        time_axis = np.arange(len(segment)) / sample_rate

        # 定义小波尺度（对应频率分辨率）
        widths = np.arange(1, 128)  # 小波尺度范围

        # 执行连续小波变换 (CWT) - 使用 ricker 小波
        coefficients = scipy.signal.cwt(segment, scipy.signal.ricker, widths)

        # 构造频率轴（近似转换）
        freq_axis = np.linspace(1, 500, len(widths))

        # ✅【关键修改1】只保留 0~500 Hz 频段
        mask = freq_axis <= 500
        coefficients = coefficients[mask, :]
        freq_axis = freq_axis[mask]

        # ✅【关键修改2】计算幅值并归一化 + 对数压缩
        magnitude = np.abs(coefficients)
        magnitude_norm = magnitude / np.max(magnitude)  # 归一化到 [0,1]
        magnitude_log = np.log(magnitude_norm + 1e-10)  # 对数压缩
        magnitude_log_norm = (magnitude_log - np.min(magnitude_log)) / (np.max(magnitude_log) - np.min(magnitude_log))

        # 更新全局最大最小值
        global_min = min(global_min, np.min(magnitude_log_norm))
        global_max = max(global_max, np.max(magnitude_log_norm))

        # 设置归一化器
        norm = Normalize(vmin=global_min, vmax=global_max)

        # 绘图 —— 使用统一归一化
        im = ax.contourf(time_axis, freq_axis, magnitude_log_norm, levels=50, cmap='viridis_r', norm=norm,
                         extend='both')

        ax.set_title(f'{class_title} ({class_code}) - 小波时频图', fontsize=16, weight='bold')
        ax.set_xlabel('时间 (s)', fontsize=12)
        ax.set_ylabel('频率 (Hz)', fontsize=12)
        ax.set_ylim([0, 500])  # 强制 y 轴范围，确保 BPFI 线可见
        ax.grid(True, linestyle='--', alpha=0.5)

        # 添加理论故障频率线
        rpm_idx = np.where(labels == class_code)[0][0]
        rpm = rpms[rpm_idx]
        theo_freqs = calculate_theoretical_frequencies(rpm)
        bpfi = theo_freqs['BPFI']
        bpfo = theo_freqs['BPFO']

        # 标注 BPFI 和 2×BPFI（确保在绘图范围内）
        if bpfi <= 500:
            ax.axhline(y=bpfi, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        if 2 * bpfi <= 500:
            ax.axhline(y=2 * bpfi, color='red', linestyle=':', linewidth=1.5, alpha=0.8)

        legend_text = []
        handles = []
        if bpfi <= 500:
            legend_text.append(f'--- BPFI={bpfi:.1f}Hz')
            handles.append(plt.Line2D([0], [0], color='red', linestyle='--', lw=1.5))
        if 2 * bpfi <= 500:
            legend_text.append(f'.... 2×BPFI={2 * bpfi:.1f}Hz')
            handles.append(plt.Line2D([0], [0], color='red', linestyle=':', lw=1.5))

        if handles:
            ax.legend(handles=handles, labels=legend_text, loc='upper right')

    # 添加颜色条（放在右侧）
    cbar = plt.colorbar(im, ax=axes.flatten().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label('归一化 log(|CWT|)', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path_cwt = os.path.join(output_dir, f'图6-小波时频图对比.png')
    plt.savefig(save_path_cwt, dpi=300)
    plt.close(fig)
    print(f"  - ✅ 小波时频图已保存至: {save_path_cwt}")


# ==============================================================================
# 1. 可视化函数 (原有代码保持不变)
# ==============================================================================
def create_final_visualizations(segments, labels, rpms, df_features, sample_rate=32000):
    """生成任务一的四张核心分析图表"""
    print("🚀 开始生成任务一全套核心可视化图表...")
    output_dir = os.path.join('..', 'data', 'processed', 'task1_visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # --- 准备工作：选取代表性样本 ---
    class_representatives = {k: np.where(labels == k)[0][0] for k in ['N', 'B', 'IR', 'OR']}
    class_titles = {'N': '正常', 'B': '滚动体故障', 'IR': '内圈故障', 'OR': '外圈故障'}
    plot_colors = {'N': 'green', 'B': 'orange', 'IR': 'blue', 'OR': 'red'}

    # --- 图一：时域波形对比图 (拆分为4个单独图) ---
    print("  - 正在生成时域波形图...")
    for class_code in ['N', 'B', 'IR', 'OR']:
        idx = class_representatives[class_code]
        segment = segments[idx]
        time_axis = np.arange(len(segment)) / sample_rate

        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, segment, color=plot_colors[class_code], linewidth=1)
        plt.title(f'{class_titles[class_code]} ({class_code}) - 时域波形图', fontsize=16, weight='bold')
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('加速度', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path1 = os.path.join(output_dir, f'图1-时域波形_{class_code}.png')
        plt.savefig(save_path1, dpi=300)
        plt.close()
    print(f"  - ✅ 图一 (时域波形图) 已保存为4个单独文件。")

    # --- 图二：频域频谱对比图 ---
    print("  - 正在生成频域频谱图...")
    freq_colors = {'BPFO': 'red', 'BPFI': 'green', 'BSF': 'blue'}
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    fig2.suptitle('四种轴承状态频域频谱对比图', fontsize=22, weight='bold')
    for ax, class_code in zip(axes2.flatten(), ['N', 'B', 'IR', 'OR']):
        idx = class_representatives[class_code]
        rpm = rpms[idx]
        segment = segments[idx]
        n = len(segment)
        freq_axis = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]
        fft_vals = np.abs(np.fft.fft(segment))[:n // 2]
        ax.plot(freq_axis, fft_vals, color=plot_colors[class_code], linewidth=1)
        ax.set_title(f'{class_titles[class_code]} ({class_code})', fontsize=16)
        ax.set_xlabel('频率 (Hz)', fontsize=12)
        ax.set_ylabel('幅值', fontsize=12)
        ax.set_xlim([0, 600])
        if class_code != 'N':
            theo_freqs = calculate_theoretical_frequencies(rpm)
            legend_handles = []
            for freq_type, freq_val in theo_freqs.items():
                color = freq_colors[freq_type]
                line = ax.axvline(x=freq_val, color=color, linestyle='--', alpha=0.9)
                legend_handles.append(line)
            ax.legend(legend_handles, [f'{k}={v:.1f}Hz' for k, v in theo_freqs.items()], loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path2 = os.path.join(output_dir, '图2-频域频谱对比.png')
    plt.savefig(save_path2, dpi=300)
    plt.close(fig2)
    print(f"  - ✅ 图二 (频域对比) 已保存。")

    # --- 图三：特征空间t-SNE降维图 ---
    print("  - 正在计算 t-SNE 降维...")
    X = df_features.drop(columns=['label', 'rpm', 'filename'])
    y_str = df_features['label']

    # === 新增：处理NaN值 ===
    print(f"    - 特征维度: {X.shape}")
    nan_count = X.isnull().sum().sum()
    print(f"    - NaN值统计: {nan_count} 个")

    # 用0填充NaN值
    X = X.fillna(0)
    print(f"    - NaN值处理完成")
    # === 新增结束 ===

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === 修改：使用max_iter替代n_iter ===
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
    tsne_results = tsne.fit_transform(X_scaled)
    # === 修改结束 ===

    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=y_str, style=y_str, s=50, alpha=0.7)
    plt.title('特征空间 t-SNE 降维可视化', fontsize=20, weight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.legend(title='故障类别')
    plt.grid(True)
    save_path3 = os.path.join(output_dir, '图3-特征空间t-SNE降维图.png')
    plt.savefig(save_path3, dpi=300)
    plt.close()
    print(f"  - ✅ 图三 (t-SNE图) 已保存。")

    # --- 图四：特征重要性排序条形图 (修改版 - 渐变色) ---
    print("  - 正在计算特征重要性...")
    feature_columns = [col for col in df_features.columns if col not in ['label', 'rpm', 'filename']]
    X_features_only = df_features[feature_columns]
    y_str = df_features['label']

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_features_only, y)
    importances = model.feature_importances_
    df_importance = pd.DataFrame({'feature': X_features_only.columns, 'importance': importances}).sort_values(
        by='importance',
        ascending=False)

    plt.figure(figsize=(12, 10))
    top_features_df = df_importance.head(20)
    features = top_features_df['feature']
    importances_vals = top_features_df['importance']
    y_positions = np.arange(len(features))

    cmap = plt.cm.get_cmap('viridis')
    norm = Normalize(vmin=importances_vals.min(), vmax=importances_vals.max())
    colors = cmap(norm(importances_vals))

    bars = plt.barh(y_positions, importances_vals, color=colors, height=0.7, edgecolor='grey', linewidth=0.5)
    plt.yticks(y_positions, features, fontsize=10)
    plt.xlabel('重要性分数', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.title('Top 20 特征重要性排序 (随机森林)', fontsize=16, weight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    cbar.set_label('重要性分数', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    save_path4 = os.path.join(output_dir, '图4-特征重要性排序.png')
    plt.savefig(save_path4, dpi=300)
    plt.close()
    print(f"  - ✅ 图四 (特征重要性图 - 渐变色) 已保存。")

    # --- 【新增】图五：时频域分析 (小波包变换 WPT) ---
    print("\n--- 新增：生成时频域分析 (小波包变换 WPT) 可视化 ---")
    visualize_wpt_energy_packets(segments, labels, rpms, sample_rate, output_dir)
    print("✅ 时频域分析 (WPT) 可视化完成。")

    # --- 【新增】图六：小波时频图 (CWT) ---
    print("\n--- 新增：生成小波时频图 (CWT) 可视化 ---")
    visualize_cwt_time_frequency(segments, labels, rpms, sample_rate, output_dir)
    print("✅ 小波时频图 (CWT) 可视化完成。")
    # --- 【新增结束】 ---


# ==============================================================================
# 2. 主程序
# ==============================================================================
if __name__ == "__main__":
    set_chinese_font()
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')

    try:
        segments = np.load(os.path.join(PROCESSED_DIR, 'source_segments.npy'))
        labels = np.load(os.path.join(PROCESSED_DIR, 'source_labels.npy'))
        rpms = np.load(os.path.join(PROCESSED_DIR, 'source_rpms.npy'))
        df_features = pd.read_csv(os.path.join(PROCESSED_DIR, 'source_features.csv'))

        print(f"成功加载所有必需数据。")
        create_final_visualizations(segments, labels, rpms, df_features)
        print("\n🎉 任务一的核心可视化图表已生成完毕！")

    except FileNotFoundError as e:
        print(f"‼️ 错误：找不到所需的数据文件 {e.filename}。请确保已完整运行01, 02, 03脚本。")
        exit()

    #     # 10_task1_final_visualizations.py
    #     import os
    #     import numpy as np
    #     import pandas as pd
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     from matplotlib import font_manager
    #     from sklearn.manifold import TSNE
    #     from sklearn.preprocessing import StandardScaler, LabelEncoder
    #     from sklearn.ensemble import RandomForestClassifier
    #     import scipy.io
    #     import scipy.signal
    #     import pywt
    #     from matplotlib.colors import Normalize
    #     from matplotlib.cm import ScalarMappable
    #
    #
    #     # ==============================================================================
    #     # 0. 辅助函数
    #     # ==============================================================================
    #     def set_chinese_font():
    #         """直接从项目文件夹加载指定的字体文件。"""
    #         font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'SourceHanSansSC-Regular.otf')
    #         if os.path.exists(font_path):
    #             font_prop = font_manager.FontProperties(fname=font_path)
    #             plt.rcParams['font.family'] = font_prop.get_name()
    #         else:
    #             plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #         plt.rcParams['axes.unicode_minus'] = False
    #         print("✅ 已设置中文字体。")
    #
    #
    #     def calculate_theoretical_frequencies(rpm):
    #         """根据转速计算SKF6205轴承的理论故障频率"""
    #         n_balls, d_ball, D_pitch = 9, 0.3126, 1.537
    #         fr = rpm / 60.0
    #         bpfo = (n_balls / 2) * fr * (1 - (d_ball / D_pitch))
    #         bpfi = (n_balls / 2) * fr * (1 + (d_ball / D_pitch))
    #         bsf = (D_pitch / (2 * d_ball)) * fr * (1 - (d_ball / D_pitch) ** 2)
    #         return {'BPFI': bpfi, 'BPFO': bpfo, 'BSF': bsf}
    #
    #
    #     # ==============================================================================
    #     # 1. 新增：时频域分析可视化函数 (小波包变换 WPT)
    #     # ==============================================================================
    #     def visualize_wpt_energy_packets(segment, sample_rate, class_title, class_code, plot_color, output_dir):
    #         """
    #         对单个信号段进行小波包变换(WPT)并可视化其能量分布。
    #         """
    #         print(f"  - 正在为 {class_title} ({class_code}) 生成小波包能量图...")
    #
    #         # 1. 执行小波包变换 (3层分解)
    #         wp = pywt.WaveletPacket(data=segment, wavelet='db1', mode='symmetric', maxlevel=3)
    #
    #         # 2. 获取第3层的所有节点
    #         nodes = wp.get_level(3, order='natural')
    #
    #         # 3. 计算每个节点的能量 (节点数据的平方和)
    #         packet_energies = np.array([np.sum(node.data ** 2) for node in nodes])
    #
    #         # 4. 创建时间轴 (与原始信号长度一致)
    #         time_axis = np.arange(len(segment)) / sample_rate
    #
    #         # 5. 绘图
    #         fig, ax = plt.subplots(figsize=(12, 6))
    #
    #         # --- 绘制原始信号 ---
    #         ax.plot(time_axis, segment, color=plot_color, alpha=0.7, linewidth=0.8, label='原始信号')
    #
    #         # --- 绘制能量包络 ---
    #         num_packets = len(packet_energies)
    #         if num_packets > 0:
    #             segment_length_per_packet = len(segment) // num_packets
    #             energy_envelope = np.repeat(packet_energies, segment_length_per_packet)
    #             if len(energy_envelope) < len(segment):
    #                 energy_envelope = np.pad(energy_envelope, (0, len(segment) - len(energy_envelope)),
    #                                          constant_values=0)
    #             elif len(energy_envelope) > len(segment):
    #                 energy_envelope = energy_envelope[:len(segment)]
    #
    #             ax.fill_between(time_axis, 0,
    #                             -energy_envelope / np.max(energy_envelope) * np.max(np.abs(segment)) * 0.3,
    #                             color=plot_color, alpha=0.4, label='WPT能量包络 (示意)')
    #
    #         ax.set_title(f'{class_title} ({class_code}) - 小波包变换 (WPT) 能量分布', fontsize=16, weight='bold')
    #         ax.set_xlabel('时间 (s)', fontsize=12)
    #         ax.set_ylabel('加速度 / 能量 (示意)', fontsize=12)
    #         ax.legend()
    #         ax.grid(True, linestyle='--', alpha=0.5)
    #         plt.tight_layout()
    #         save_path_wpt = os.path.join(output_dir, f'图5-WPT能量分布_{class_code}.png')
    #         plt.savefig(save_path_wpt, dpi=300)
    #         plt.close(fig)
    #         print(f"  - ✅ {class_title} ({class_code}) 的 WPT 能量图已保存至: {save_path_wpt}")
    #
    #
    #     # ==============================================================================
    #     # 1. 新增：小波时频图（CWT）可视化函数 —— 【已修正】
    #     # ==============================================================================
    #     def visualize_cwt_time_frequency(segment, sample_rate, class_title, class_code, plot_color, output_dir):
    #         """
    #         使用连续小波变换（CWT）生成时频图，并标注理论故障频率。
    #         """
    #         print(f"  - 正在为 {class_title} ({class_code}) 生成小波时频图...")
    #
    #         # 时间轴
    #         time_axis = np.arange(len(segment)) / sample_rate
    #
    #         # 定义小波尺度（对应频率分辨率）
    #         widths = np.arange(1, 128)  # 小波尺度范围
    #
    #         # 执行连续小波变换 (CWT) - 使用 ricker 小波
    #         coefficients = scipy.signal.cwt(segment, scipy.signal.ricker, widths)
    #
    #         # 构造频率轴（近似转换）
    #         freq_axis = np.linspace(1, 500, len(widths))
    #
    #         # ✅【关键修改1】只保留 0~500 Hz 频段
    #         mask = freq_axis <= 500
    #         coefficients = coefficients[mask, :]
    #         freq_axis = freq_axis[mask]
    #
    #         # ✅【关键修改2】计算幅值并归一化 + 对数压缩
    #         magnitude = np.abs(coefficients)
    #         magnitude_norm = magnitude / np.max(magnitude)  # 归一化到 [0,1]
    #         magnitude_log = np.log(magnitude_norm + 1e-10)  # 对数压缩
    #         magnitude_log_norm = (magnitude_log - np.min(magnitude_log)) / (
    #                     np.max(magnitude_log) - np.min(magnitude_log))
    #
    #         # 绘图
    #         fig, ax = plt.subplots(figsize=(12, 8))
    #
    #         # ✅【关键修改3】绘制归一化对数幅值，使用 'viridis_r' 色图（紫→蓝→绿→黄）
    #         im = ax.contourf(time_axis, freq_axis, magnitude_log_norm, levels=50, cmap='viridis_r', extend='both')
    #
    #         cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    #         cbar.set_label('归一化 log(|CWT|)', rotation=270, labelpad=20, fontsize=12)
    #
    #         ax.set_title(f'{class_title} ({class_code}) - 小波时频图', fontsize=16, weight='bold')
    #         ax.set_xlabel('时间 (s)', fontsize=12)
    #         ax.set_ylabel('频率 (Hz)', fontsize=12)
    #         ax.set_ylim([0, 500])  # 强制 y 轴范围，确保 BPFI 线可见
    #         ax.grid(True, linestyle='--', alpha=0.5)
    #
    #         # 添加理论故障频率线
    #         idx = np.where(labels == class_code)[0][0]  # 假设 labels 是全局变量
    #         rpm = rpms[idx]
    #         theo_freqs = calculate_theoretical_frequencies(rpm)
    #         bpfi = theo_freqs['BPFI']
    #         bpfo = theo_freqs['BPFO']
    #
    #         # 标注 BPFI 和 2×BPFI（确保在绘图范围内）
    #         if bpfi <= 500:
    #             ax.axhline(y=bpfi, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    #         if 2 * bpfi <= 500:
    #             ax.axhline(y=2 * bpfi, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
    #
    #         legend_text = []
    #         handles = []
    #         if bpfi <= 500:
    #             legend_text.append(f'--- BPFI={bpfi:.1f}Hz')
    #             handles.append(plt.Line2D([0], [0], color='red', linestyle='--', lw=1.5))
    #         if 2 * bpfi <= 500:
    #             legend_text.append(f'.... 2×BPFI={2 * bpfi:.1f}Hz')
    #             handles.append(plt.Line2D([0], [0], color='red', linestyle=':', lw=1.5))
    #
    #         if handles:
    #             ax.legend(handles=handles, labels=legend_text, loc='upper right')
    #
    #         plt.tight_layout()
    #         save_path_cwt = os.path.join(output_dir, f'图6-小波时频图_{class_code}.png')
    #         plt.savefig(save_path_cwt, dpi=300)
    #         plt.close(fig)
    #         print(f"  - ✅ {class_title} ({class_code}) 的小波时频图已保存至: {save_path_cwt}")
    #
    #
    #     # ==============================================================================
    #     # 1. 可视化函数 (原有代码保持不变)
    #     # ==============================================================================
    #     def create_final_visualizations(segments, labels, rpms, df_features, sample_rate=32000):
    #         """生成任务一的四张核心分析图表"""
    #         print("🚀 开始生成任务一全套核心可视化图表...")
    #         output_dir = os.path.join('..', 'data', 'processed', 'task1_visualizations')
    #         os.makedirs(output_dir, exist_ok=True)
    #
    #         # --- 准备工作：选取代表性样本 ---
    #         class_representatives = {k: np.where(labels == k)[0][0] for k in ['N', 'B', 'IR', 'OR']}
    #         class_titles = {'N': '正常', 'B': '滚动体故障', 'IR': '内圈故障', 'OR': '外圈故障'}
    #         plot_colors = {'N': 'green', 'B': 'orange', 'IR': 'blue', 'OR': 'red'}
    #
    #         # --- 图一：时域波形对比图 ---
    #         fig1, axes1 = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
    #         fig1.suptitle('四种轴承状态时域波形对比图', fontsize=22, weight='bold')
    #         for ax, class_code in zip(axes1.flatten(), ['N', 'B', 'IR', 'OR']):
    #             idx = class_representatives[class_code]
    #             segment = segments[idx]
    #             time_axis = np.arange(len(segment)) / sample_rate
    #             ax.plot(time_axis, segment, color=plot_colors[class_code], linewidth=1)
    #             ax.set_title(f'{class_titles[class_code]} ({class_code})', fontsize=16)
    #             ax.set_xlabel('时间 (s)', fontsize=12)
    #             ax.set_ylabel('加速度', fontsize=12)
    #             ax.grid(True, linestyle='--', alpha=0.6)
    #         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #         save_path1 = os.path.join(output_dir, '图1-时域波形对比.png')
    #         plt.savefig(save_path1, dpi=300)
    #         plt.close(fig1)
    #         print(f"  - ✅ 图一 (时域对比) 已保存。")
    #
    #         # --- 图二：频域频谱对比图 ---
    #         fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    #         fig2.suptitle('四种轴承状态频域频谱对比图', fontsize=22, weight='bold')
    #         freq_colors = {'BPFO': 'red', 'BPFI': 'green', 'BSF': 'blue'}
    #         for ax, class_code in zip(axes2.flatten(), ['N', 'B', 'IR', 'OR']):
    #             idx = class_representatives[class_code]
    #             rpm = rpms[idx]
    #             segment = segments[idx]
    #             n = len(segment)
    #             freq_axis = np.fft.fftfreq(n, 1 / sample_rate)[:n // 2]
    #             fft_vals = np.abs(np.fft.fft(segment))[:n // 2]
    #             ax.plot(freq_axis, fft_vals, color=plot_colors[class_code], linewidth=1)
    #             ax.set_title(f'{class_titles[class_code]} ({class_code})', fontsize=16)
    #             ax.set_xlabel('频率 (Hz)', fontsize=12)
    #             ax.set_ylabel('幅值', fontsize=12)
    #             ax.set_xlim([0, 600])
    #             if class_code != 'N':
    #                 theo_freqs = calculate_theoretical_frequencies(rpm)
    #                 legend_handles = []
    #                 for freq_type, freq_val in theo_freqs.items():
    #                     color = freq_colors[freq_type]
    #                     line = ax.axvline(x=freq_val, color=color, linestyle='--', alpha=0.9)
    #                     legend_handles.append(line)
    #                 ax.legend(legend_handles, [f'{k}={v:.1f}Hz' for k, v in theo_freqs.items()], loc='upper right')
    #         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #         save_path2 = os.path.join(output_dir, '图2-频域频谱对比.png')
    #         plt.savefig(save_path2, dpi=300)
    #         plt.close(fig2)
    #         print(f"  - ✅ 图二 (频域对比) 已保存。")
    #
    #         # --- 图三：特征空间t-SNE降维图 ---
    #         print("  - 正在计算 t-SNE 降维...")
    #         X = df_features.drop(columns=['label', 'rpm', 'filename'])
    #         y_str = df_features['label']
    #
    #         # === 新增：处理NaN值 ===
    #         print(f"    - 特征维度: {X.shape}")
    #         nan_count = X.isnull().sum().sum()
    #         print(f"    - NaN值统计: {nan_count} 个")
    #
    #         # 用0填充NaN值
    #         X = X.fillna(0)
    #         print(f"    - NaN值处理完成")
    #         # === 新增结束 ===
    #
    #         scaler = StandardScaler()
    #         X_scaled = scaler.fit_transform(X)
    #
    #         # === 修改：使用max_iter替代n_iter ===
    #         tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
    #         tsne_results = tsne.fit_transform(X_scaled)
    #         # === 修改结束 ===
    #
    #         plt.figure(figsize=(12, 10))
    #         sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=y_str, style=y_str, s=50, alpha=0.7)
    #         plt.title('特征空间 t-SNE 降维可视化', fontsize=20, weight='bold')
    #         plt.xlabel('t-SNE Component 1', fontsize=14)
    #         plt.ylabel('t-SNE Component 2', fontsize=14)
    #         plt.legend(title='故障类别')
    #         plt.grid(True)
    #         save_path3 = os.path.join(output_dir, '图3-特征空间t-SNE降维图.png')
    #         plt.savefig(save_path3, dpi=300)
    #         plt.close()
    #         print(f"  - ✅ 图三 (t-SNE图) 已保存。")
    #
    #         # --- 图四：特征重要性排序条形图 (修改版 - 渐变色) ---
    #         print("  - 正在计算特征重要性...")
    #         feature_columns = [col for col in df_features.columns if col not in ['label', 'rpm', 'filename']]
    #         X_features_only = df_features[feature_columns]
    #         y_str = df_features['label']
    #
    #         le = LabelEncoder()
    #         y = le.fit_transform(y_str)
    #         model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    #         model.fit(X_features_only, y)
    #         importances = model.feature_importances_
    #         df_importance = pd.DataFrame({'feature': X_features_only.columns, 'importance': importances}).sort_values(
    #             by='importance',
    #             ascending=False)
    #
    #         plt.figure(figsize=(12, 10))
    #         top_features_df = df_importance.head(20)
    #         features = top_features_df['feature']
    #         importances_vals = top_features_df['importance']
    #         y_positions = np.arange(len(features))
    #
    #         cmap = plt.cm.get_cmap('viridis')
    #         norm = Normalize(vmin=importances_vals.min(), vmax=importances_vals.max())
    #         colors = cmap(norm(importances_vals))
    #
    #         bars = plt.barh(y_positions, importances_vals, color=colors, height=0.7, edgecolor='grey', linewidth=0.5)
    #         plt.yticks(y_positions, features, fontsize=10)
    #         plt.xlabel('重要性分数', fontsize=12)
    #         plt.ylabel('特征名称', fontsize=12)
    #         plt.title('Top 20 特征重要性排序 (随机森林)', fontsize=16, weight='bold')
    #         plt.gca().invert_yaxis()
    #         plt.grid(axis='x', linestyle='--', alpha=0.6)
    #
    #         sm = ScalarMappable(cmap=cmap, norm=norm)
    #         sm.set_array([])
    #         cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
    #         cbar.set_label('重要性分数', rotation=270, labelpad=20, fontsize=12)
    #
    #         plt.tight_layout()
    #         save_path4 = os.path.join(output_dir, '图4-特征重要性排序.png')
    #         plt.savefig(save_path4, dpi=300)
    #         plt.close()
    #         print(f"  - ✅ 图四 (特征重要性图 - 渐变色) 已保存。")
    #
    #         # --- 【新增】图五：时频域分析 (小波包变换 WPT) ---
    #         print("\n--- 新增：生成时频域分析 (小波包变换 WPT) 可视化 ---")
    #         for class_code in ['N', 'B', 'IR', 'OR']:
    #             idx = class_representatives[class_code]
    #             segment = segments[idx]
    #             visualize_wpt_energy_packets(
    #                 segment, sample_rate, class_titles[class_code], class_code, plot_colors[class_code], output_dir
    #             )
    #         print("✅ 时频域分析 (WPT) 可视化完成。")
    #
    #         # --- 【新增】图六：小波时频图 (CWT) ---
    #         print("\n--- 新增：生成小波时频图 (CWT) 可视化 ---")
    #         for class_code in ['N', 'B', 'IR', 'OR']:
    #             idx = class_representatives[class_code]
    #             segment = segments[idx]
    #             visualize_cwt_time_frequency(
    #                 segment, sample_rate, class_titles[class_code], class_code, plot_colors[class_code], output_dir
    #             )
    #         print("✅ 小波时频图 (CWT) 可视化完成。")
    #         # --- 【新增结束】 ---
    #
    #
    #     # ==============================================================================
    #     # 2. 主程序
    #     # ==============================================================================
    #     if __name__ == "__main__":
    #         set_chinese_font()
    #         PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    #
    #         try:
    #             segments = np.load(os.path.join(PROCESSED_DIR, 'source_segments.npy'))
    #             labels = np.load(os.path.join(PROCESSED_DIR, 'source_labels.npy'))
    #             rpms = np.load(os.path.join(PROCESSED_DIR, 'source_rpms.npy'))
    #             df_features = pd.read_csv(os.path.join(PROCESSED_DIR, 'source_features.csv'))
    #
    #             print(f"成功加载所有必需数据。")
    #             create_final_visualizations(segments, labels, rpms, df_features)
    #             print("\n🎉 任务一的核心可视化图表已生成完毕！")
    #
    #         except FileNotFoundError as e:
    #             print(f"‼️ 错误：找不到所需的数据文件 {e.filename}。请确保已完整运行01, 02, 03脚本。")
    #             exit()
    #
    #