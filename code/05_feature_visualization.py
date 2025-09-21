import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


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
# 1. 核心可视化函数 (生成全部三种图表)
# ==============================================================================
def create_comprehensive_feature_visualizations(df_features):
    """
    【最终版】生成一套完整的特征分析图表：箱线图、t-SNE降维图、相关性热力图。
    """
    print("🚀 开始生成全套特征分析图表...")
    output_dir = os.path.join('..', 'data', 'processed')
    class_order = ['N', 'B', 'IR', 'OR']

    # --- 图表一：特征分布箱线图 (合并版) ---
    features_to_plot = ['kurtosis', 'rms', 'BPFO_1x']
    fig1, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig1.suptitle('关键特征在不同故障类别下的分布对比', fontsize=22, weight='bold')

    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        sns.boxplot(data=df_features, x='label', y=feature, ax=ax, order=class_order)
        ax.set_title(f'特征: "{feature}"', fontsize=16)
        ax.set_xlabel('故障类别', fontsize=12)
        ax.set_ylabel('特征值', fontsize=12)
        if feature == 'kurtosis':
            ax.set_yscale('log')
            ax.set_ylabel('特征值 (对数坐标)', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path1 = os.path.join(output_dir, '特征分布对比图(箱线图).png')
    plt.savefig(save_path1, dpi=300)
    print(f"  - ✅ 图表一 (箱线图) 已保存至: {os.path.abspath(save_path1)}")
    plt.close(fig1)

    # --- 图表二：t-SNE 降维散点图 ---
    print("  - 正在计算 t-SNE 降维，这可能需要一点时间...")
    features = df_features.drop(columns=['label', 'rpm'])
    labels = df_features['label']

    # 数据标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 执行 t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_results = tsne.fit_transform(features_scaled)

    # 绘图
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=labels,
        palette=sns.color_palette("hls", 4),
        style=labels,
        s=50,  # 调整点的大小
        alpha=0.7
    )
    plt.title('特征空间 t-SNE 降维可视化', fontsize=20, weight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.legend(title='故障类别', fontsize=12)
    plt.grid(True)

    save_path2 = os.path.join(output_dir, '特征空间t-SNE降维图.png')
    plt.savefig(save_path2, dpi=300)
    print(f"  - ✅ 图表二 (t-SNE图) 已保存至: {os.path.abspath(save_path2)}")
    plt.close()

    # --- 图表三：特征相关性热力图 ---
    plt.figure(figsize=(18, 15))
    corr_matrix = features.corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)  # annot=False 因为特征太多，显示数值会很乱
    plt.title('特征相关性热力图', fontsize=20, weight='bold')

    save_path3 = os.path.join(output_dir, '特征相关性热力图.png')
    plt.savefig(save_path3, dpi=300)
    print(f"  - ✅ 图表三 (热力图) 已保存至: {os.path.abspath(save_path3)}")
    plt.close()


# ==============================================================================
# 2. 主程序
# ==============================================================================
if __name__ == "__main__":
    set_chinese_font()
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features.csv')

    try:
        df_features = pd.read_csv(FEATURES_PATH)
        print(f"成功加载特征集: {df_features.shape[0]} 个样本, {df_features.shape[1]} 个特征。")
        create_comprehensive_feature_visualizations(df_features)
        print("\n🎉 任务一的全套可视化分析工作已完成！")
    except FileNotFoundError:
        print(f"‼️ 错误：找不到特征文件 {FEATURES_PATH}。请先运行 03_feature_extraction.py。")
        exit()