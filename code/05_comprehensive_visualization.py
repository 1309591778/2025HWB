import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier


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
# 1. 可视化函数
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

    # --- 图一：时域波形对比图 ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
    fig1.suptitle('四种轴承状态时域波形对比图', fontsize=22, weight='bold')
    for ax, class_code in zip(axes1.flatten(), ['N', 'B', 'IR', 'OR']):
        # ... (代码与之前版本相同)
        idx = class_representatives[class_code]
        segment = segments[idx]
        time_axis = np.arange(len(segment)) / sample_rate
        ax.plot(time_axis, segment, color=plot_colors[class_code], linewidth=1)
        ax.set_title(f'{class_titles[class_code]} ({class_code})', fontsize=16)
        ax.set_xlabel('时间 (s)', fontsize=12)
        ax.set_ylabel('加速度', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path1 = os.path.join(output_dir, '图1-时域波形对比.png')
    plt.savefig(save_path1, dpi=300)
    plt.close(fig1)
    print(f"  - ✅ 图一 (时域对比) 已保存。")

    # --- 图二：频域频谱对比图 ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    fig2.suptitle('四种轴承状态频域频谱对比图', fontsize=22, weight='bold')
    freq_colors = {'BPFO': 'red', 'BPFI': 'green', 'BSF': 'blue'}
    for ax, class_code in zip(axes2.flatten(), ['N', 'B', 'IR', 'OR']):
        # ... (代码与之前版本相同)
        idx, rpm = class_representatives[class_code], rpms[class_representatives[class_code]]
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

    # --- 图四：特征重要性排序条形图 ---
    print("  - 正在计算特征重要性...")
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)  # 注意：此处使用未缩放的X，对于树模型影响不大
    importances = model.feature_importances_
    df_importance = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(by='importance',
                                                                                                ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=df_importance.head(20))
    plt.title('Top 20 特征重要性排序 (随机森林)', fontsize=16)
    plt.xlabel('重要性分数')
    plt.ylabel('特征名称')
    plt.grid(True)
    plt.tight_layout()
    save_path4 = os.path.join(output_dir, '图4-特征重要性排序.png')
    plt.savefig(save_path4, dpi=300)
    plt.close()
    print(f"  - ✅ 图四 (特征重要性图) 已保存。")


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
        # 加载包含所有特征的文件，用于t-SNE和重要性分析
        df_features = pd.read_csv(os.path.join(PROCESSED_DIR, 'source_features.csv'))

        print(f"成功加载所有必需数据。")
        create_final_visualizations(segments, labels, rpms, df_features)
        print("\n🎉 任务一的核心可视化图表已生成完毕！")

    except FileNotFoundError as e:
        print(f"‼️ 错误：找不到所需的数据文件 {e.filename}。请确保已完整运行01, 02, 03脚本。")
        exit()