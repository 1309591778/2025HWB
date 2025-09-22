import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import xgboost as xgb
# 【核心】导入GroupShuffleSplit，不再使用train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import joblib


# (字体设置函数 set_chinese_font 保持不变)
def set_chinese_font():
    font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'SourceHanSansSC-Regular.otf')
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")


if __name__ == "__main__":
    set_chinese_font()

    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    # 【注意】确保使用的是经过特征筛选后的最优特征集
    FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features_selected.csv')
    OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'task2_outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🚀 步骤 1: 加载数据...")
    try:
        df_features = pd.read_csv(FEATURES_PATH)
    except FileNotFoundError:
        print(f"‼️ 错误：找不到特征文件 {FEATURES_PATH}。请先运行任务一的脚本。")
        exit()

    # --- 数据准备 ---
    if 'filename' not in df_features.columns:
        print("‼️ 错误：特征文件中缺少'filename'列。请按指导修改并重新运行02和03脚本。")
        exit()

    X_raw = df_features.drop(columns=['label', 'rpm', 'filename'])
    y_str = df_features['label']
    groups = df_features['filename']

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    print("✅ 数据加载和标签编码完成。")

    # --- 步骤 2: 使用分组划分(GroupShuffleSplit)，从根源避免泄露 ---
    print("\n🚀 步骤 2: 使用分组划分训练集与测试集...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    # groups参数是关键，它确保了来自同一个文件的样本不会同时出现在训练集和测试集
    train_idx, test_idx = next(gss.split(X_raw, y, groups))

    X_train_raw, X_test_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    print(f"训练集包含 {len(train_groups)} 个独立文件。")
    print(f"测试集包含 {len(test_groups)} 个独立文件。")
    print(f"训练集和测试集文件重叠数量: {len(train_groups.intersection(test_groups))}")  # 结果应为0
    print(f"✅ 划分完成。训练集样本数: {len(X_train_raw)}, 测试集样本数: {len(X_test_raw)}")

    # --- 步骤 3: 进行特征缩放 (正确流程：先划分，后缩放) ---
    print("\n🚀 步骤 3: 进行特征缩放...")
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    print("✅ 特征缩放完成。")

    # --- 步骤 4: 模型训练 (XGBoost) ---
    print("\n🚀 步骤 4: 训练XGBoost模型...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False,
                              random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print("✅ 模型训练完成。")

    # (后续的评估、特征重要性、保存代码与你提供的版本基本一致，此处为完整版)
    print("\n🚀 步骤 5: 评估模型性能...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 总体准确率 (无数据泄露): {accuracy:.4f}")

    class_names = le.classes_
    print("\n📊 分类报告:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('源域诊断模型混淆矩阵 (无数据泄露)', fontsize=16)
    plt.ylabel('真实类别', fontsize=12)
    plt.xlabel('预测类别', fontsize=12)
    conf_matrix_path = os.path.join(OUTPUT_DIR, '任务二-源域诊断混淆矩阵(无泄露).png')
    plt.savefig(conf_matrix_path, dpi=300)
    print(f"✅ 混淆矩阵已保存至: {os.path.abspath(conf_matrix_path)}")
    plt.close()

    print("\n🚀 步骤 6: 分析特征重要性...")
    importances = model.feature_importances_
    df_importance = pd.DataFrame({'feature': X_raw.columns, 'importance': importances}).sort_values(by='importance',
                                                                                                    ascending=False)
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=df_importance)
    plt.title('特征重要性排序', fontsize=16)
    plt.xlabel('重要性分数', fontsize=12)
    plt.ylabel('特征名称', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    feature_importance_path = os.path.join(OUTPUT_DIR, '任务二-特征重要性排序.png')
    plt.savefig(feature_importance_path, dpi=300)
    print(f"✅ 特征重要性图已保存至: {os.path.abspath(feature_importance_path)}")
    plt.close()

    print("\n🚀 步骤 7: 保存模型...")
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'xgb_model.joblib'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
    joblib.dump(le, os.path.join(OUTPUT_DIR, 'label_encoder.joblib'))
    print(f"✅ 模型、缩放器和标签编码器已保存至: {os.path.abspath(OUTPUT_DIR)}")

    print("\n🎉 任务二：源域故障诊断全部工作已完成！")