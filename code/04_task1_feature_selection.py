import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter  # 【新增】导入用于格式化百分比的工具
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# ==============================================================================
# 0. 字体设置函数
# ==============================================================================
def set_chinese_font():
    """
    强制设置中文字体为 'Microsoft YaHei'，解决中文显示问题。
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")


# ==============================================================================
# 1. 主程序
# ==============================================================================
if __name__ == "__main__":
    set_chinese_font()

    # --- 路径定义 ---
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features.csv')
    OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'task1_outputs')  # 为任务一的输出创建一个专门的文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 数据加载 ---
    try:
        df_features = pd.read_csv(FEATURES_PATH)
    except FileNotFoundError:
        print(f"‼️ 错误：找不到特征文件 {FEATURES_PATH}。请先运行03脚本生成增强版特征集。")
        exit()

    if 'filename' not in df_features.columns:
        print("‼️ 错误：特征文件中缺少'filename'列。请按指导修改并重新运行02和03脚本。")
        exit()

    X = df_features.drop(columns=['label', 'rpm', 'filename'])
    y_str = df_features['label']
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # --- 后验验证 1: 相关性分析 ---
    print("\n🚀 后验验证 1: 分析高度相关的特征对...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated = [column for column in upper.columns if any(upper[column] > 0.8)]
    for col in highly_correlated:
        correlated_with = upper.index[upper[col] > 0.8].tolist()
        print(f"  - 特征 '{col}' 与 {correlated_with} 高度相关 (r>0.8)")

    # --- 后验验证 2: 方差分析 (ANOVA) ---
    print("\n🚀 后验验证 2: ANOVA F值排序 (验证特征区分度)...")
    f_values, p_values = f_classif(X, y)
    df_anova = pd.DataFrame({'feature': X.columns, 'F_score': f_values}).sort_values(by='F_score', ascending=False)
    print("  - ANOVA F值排名前15的特征:")
    print(df_anova.head(15))

    # --- 后验验证 3: 随机森林特征重要性 ---
    print("\n🚀 后验验证 3: 随机森林特征重要性排序...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    importances = model.feature_importances_
    df_importance = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values(by='importance',
                                                                                                ascending=False)
    print("  - 随机森林模型认为最重要的15个特征:")
    print(df_importance.head(15))

    # 可视化特征重要性条形图
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=df_importance.head(20))
    plt.title('Top 20 特征重要性排序 (随机森林)', fontsize=16)
    save_path_bar = os.path.join(OUTPUT_DIR, 'task1_feature_importance_bar.png')
    plt.savefig(save_path_bar, dpi=300)
    print(f"\n✅ 特征重要性排序图已保存至: {os.path.abspath(save_path_bar)}")
    plt.close()

    # --- 【新增】可视化：特征累积重要性曲线 ---
    print("\n🚀 新增可视化：生成特征累积重要性曲线...")

    # 计算累积重要性分数 (已经是按重要性降序排列的)
    cumulative_importance = np.cumsum(df_importance['importance'])

    # 创建图表
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, marker='o', linestyle='--')

    # 设置Y轴为百分比格式
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    # 标记95%和99%的重要性阈值线
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% 累积重要性')

    # 找到达到95%阈值需要多少个特征
    try:
        num_features_95 = np.where(cumulative_importance >= 0.95)[0][0] + 1
        plt.text(num_features_95 + 0.5, 0.93, f'Top {num_features_95} 个特征', color='r', fontsize=12)
        # 绘制垂线以更清晰地指示位置
        plt.axvline(x=num_features_95, color='r', linestyle=':', alpha=0.7)
    except IndexError:
        print("  - 注意：所有特征的累积重要性未达到95%。")

    plt.title('特征累积重要性曲线', fontsize=18, weight='bold')
    plt.xlabel('按重要性排序的特征数量', fontsize=14)
    plt.ylabel('累积重要性', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path_curve = os.path.join(OUTPUT_DIR, 'task1_cumulative_feature_importance_curve.png')
    plt.savefig(save_path_curve, dpi=300)
    print(f"✅ 特征累积重要性曲线图已保存至: {os.path.abspath(save_path_curve)}")
    plt.close()

    # --- 最终筛选决策 ---
    print("\n🚀 最终决策：筛选特征...")
    # 我们可以根据累积重要性曲线的结果来动态决定保留多少个特征
    # 例如，保留达到95%重要性的所有特征
    final_features_to_keep = df_importance['feature'].iloc[:num_features_95].tolist()

    # 检查并移除高度相关的特征
    # (这是一个简化的去冗余逻辑示例)
    for col in highly_correlated:
        if col in final_features_to_keep:
            correlated_with = upper.index[upper[col] > 0.8].tolist()
            # 如果与col相关的特征也在保留列表中，且col的重要性更低，则考虑移除col
            # 此处逻辑较为复杂，可以先手动决策，例如我们已知rms和std_dev高度相关
            if 'std_dev' in final_features_to_keep and 'rms' in final_features_to_keep:
                # 假设我们通过ANOVA或模型重要性，决定保留std_dev
                final_features_to_keep.remove('rms')

    print(f"  - ✅ 决策完成：筛选出 {len(final_features_to_keep)} 个特征用于后续建模。")
    print(f"  - 筛选出的特征列表: {final_features_to_keep}")

    # 保存最终筛选的特征集
    final_columns = ['filename', 'label', 'rpm'] + final_features_to_keep
    df_final_source = df_features[final_columns]

    save_path_source = os.path.join(PROCESSED_DIR, 'source_features_selected.csv')
    df_final_source.to_csv(save_path_source, index=False)
    print(f"  - ✅ 筛选后的源域特征集已保存至: {os.path.abspath(save_path_source)}")

    # 保存筛选出的特征名称列表，供06脚本使用
    save_path_feature_list = os.path.join(PROCESSED_DIR, 'selected_feature_names.txt')
    with open(save_path_feature_list, 'w') as f:
        for feature_name in final_features_to_keep:
            f.write(f"{feature_name}\n")
    print(f"  - ✅ 筛选出的特征名称列表已保存至: {os.path.abspath(save_path_feature_list)}")

    print("\n🎉 任务一的特征筛选工作已完成！")