import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import xgboost as xgb
# 【注意】我们现在只使用 train_test_split，因为这个版本的代码不处理分组问题
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import joblib


# ==============================================================================
# 0. 字体设置函数 (保持不变)
# ==============================================================================
def set_chinese_font():
    """直接从项目文件夹加载指定的字体文件。"""
    # 为了让代码直接运行，我们先注释掉字体加载，你可以根据需要恢复
    # font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'SourceHanSansSC-Regular.otf')
    # if os.path.exists(font_path):
    #     font_prop = font_manager.FontProperties(fname=font_path)
    #     plt.rcParams['font.family'] = font_prop.get_name()
    # else:
    #     print(f"‼️ 警告：找不到字体文件 {font_path}，建议下载。")
    #     plt.rcParams['font.sans-serif'] = ['sans-serif']

    # 采用一个更通用的设置
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
    # 【注意】确保你使用的是增强了特征、扩充了数据量的最新版特征文件
    FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features_selected.csv')
    OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'task2_outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. 数据加载与准备 ---
    print("🚀 步骤 1: 加载数据...")
    try:
        df_features = pd.read_csv(FEATURES_PATH)
    except FileNotFoundError:
        print(f"‼️ 错误：找不到特征文件 {FEATURES_PATH}。请先重新运行任务一的脚本。")
        exit()

    # 从特征文件中分离出特征X和标签y
    # 我们假设'filename'列存在并需要被移除
    if 'filename' in df_features.columns:
        X = df_features.drop(columns=['label', 'rpm', 'filename'])
    else:
        X = df_features.drop(columns=['label', 'rpm'])

    y_str = df_features['label']

    # 标签编码 (将'N','B','IR','OR'转为0,1,2,3)
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    print("✅ 数据加载和标签编码完成。")

    # --- 【核心修改】步骤2: 先划分训练集与测试集 ---
    print("\n🚀 步骤 2: 划分训练集与测试集 (在缩放前进行)...")
    # 我们现在对原始的、未缩放的特征X进行划分
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y  # 必须进行分层抽样
    )
    print(f"✅ 划分完成。训练集样本数: {len(X_train_raw)}, 测试集样本数: {len(X_test_raw)}")

    # --- 【核心修改】步骤3: 后进行特征缩放 ---
    print("\n🚀 步骤 3: 进行特征缩放 (正确流程)...")
    scaler = StandardScaler()

    # 只用训练集的数据来'训练'缩放器 (fit)
    scaler.fit(X_train_raw)

    # 用这个训练好的缩放器，分别对训练集和测试集进行转换 (transform)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    print("✅ 特征缩放完成。")

    # --- 步骤 4: 模型训练 (XGBoost) ---
    print("\n🚀 步骤 4: 训练XGBoost模型...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("✅ 模型训练完成。")

    # --- 步骤 5: 模型评估 ---
    print("\n🚀 步骤 5: 评估模型性能...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 总体准确率 (修正后): {accuracy:.4f}")

    class_names = le.classes_
    print("\n📊 分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('源域诊断模型混淆矩阵 (修正后)', fontsize=16)
    plt.ylabel('真实类别', fontsize=12)
    plt.xlabel('预测类别', fontsize=12)
    conf_matrix_path = os.path.join(OUTPUT_DIR, '任务二-源域诊断混淆矩阵(修正后).png')
    plt.savefig(conf_matrix_path, dpi=300)
    print(f"✅ 混淆矩阵已保存至: {os.path.abspath(conf_matrix_path)}")
    plt.close()

    # --- 步骤 6: 特征重要性分析 ---
    print("\n🚀 步骤 6: 分析特征重要性...")
    importances = model.feature_importances_
    df_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
    df_importance = df_importance.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(12, 12))
    sns.barplot(x='importance', y='feature', data=df_importance.head(30))
    plt.title('Top 30 特征重要性排序', fontsize=16)
    plt.xlabel('重要性分数', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    feature_importance_path = os.path.join(OUTPUT_DIR, '任务二-特征重要性排序.png')
    plt.savefig(feature_importance_path, dpi=300)
    print(f"✅ 特征重要性图已保存至: {os.path.abspath(feature_importance_path)}")
    plt.close()

    # --- 步骤 7: 保存模型及预处理器 ---
    print("\n🚀 步骤 7: 保存模型...")
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'xgb_model.joblib'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    joblib.dump(le, os.path.join(OUTPUT_DIR, 'label_encoder.joblib'))
    print(f"✅ 模型、缩放器和标签编码器已保存至: {os.path.abspath(OUTPUT_DIR)}")

    print("\n🎉 任务二：源域故障诊断全部工作已完成！")