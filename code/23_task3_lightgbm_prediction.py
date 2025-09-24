# 23_task3_lightgbm_transfer.py
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_output_dir():
    output_dir = os.path.join('..', 'data', 'processed', 'task3_outputs_lightgbm')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    print("🚀 任务三：基于 LightGBM 的迁移诊断（无 DANN）开始执行...")
    output_dir = create_output_dir()

    # --- 阶段一：加载预处理器和模型 ---
    print("\n--- 阶段一：加载预处理器和模型 ---")
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    TASK2_OUTPUTS_DIR = os.path.join(PROCESSED_DIR, 'task2_outputs_final')

    # 加载预处理器
    SCALER_PATH = os.path.join(TASK2_OUTPUTS_DIR, 'final_scaler.joblib')
    LE_PATH = os.path.join(TASK2_OUTPUTS_DIR, 'final_label_encoder.joblib')
    MODEL_PATH = os.path.join(TASK2_OUTPUTS_DIR, 'final_lgb_model.txt')

    try:
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LE_PATH)
        lgb_model = lgb.Booster(model_file=MODEL_PATH)
        print("✅ 成功加载 StandardScaler、LabelEncoder 和 LightGBM 模型。")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        exit(1)

    # --- 阶段二：加载目标域特征 ---
    print("\n--- 阶段二：加载目标域特征 ---")
    TARGET_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'target_features.csv')
    try:
        df_target = pd.read_csv(TARGET_FEATURES_PATH)
        print(f"✅ 成功加载目标域特征: {df_target.shape}")
    except Exception as e:
        print(f"❌ 无法加载目标域特征: {e}")
        exit(1)

    # 提取特征列（排除 'source_file', 'rpm'）
    feature_cols = [col for col in df_target.columns if col not in ['source_file', 'rpm']]
    X_target = df_target[feature_cols].values
    filenames = df_target['source_file'].values

    # 标准化
    X_target_scaled = scaler.transform(X_target)

    # --- 阶段三：预测 ---
    print("\n--- 阶段三：执行预测 ---")
    y_pred_proba = lgb_model.predict(X_target_scaled)
    y_pred_int = np.argmax(y_pred_proba, axis=1)
    y_pred_labels = le.inverse_transform(y_pred_int)
    confidence = np.max(y_pred_proba, axis=1)

    print("📊 预测类别分布（样本级）:")
    unique, counts = np.unique(y_pred_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c} 个样本")

    # --- 阶段四：按文件名投票 ---
    print("\n--- 阶段四：按文件名进行投票 ---")
    file_predictions = {}
    for filename in np.unique(filenames):
        mask = (filenames == filename)
        file_labels = y_pred_labels[mask]
        file_conf = confidence[mask]

        # 投票：得票最多的类别
        votes = pd.Series(file_labels).value_counts()
        predicted_class = votes.index[0]
        vote_ratio = votes.iloc[0] / len(file_labels)
        avg_confidence = np.mean(file_conf)
        final_confidence = vote_ratio * avg_confidence  # 综合置信度

        file_predictions[filename] = {
            'predicted_label': predicted_class,
            'confidence': final_confidence,
            'total_samples': len(file_labels),
            'vote_distribution': votes.to_dict()
        }
        print(f"  - {filename}: {predicted_class} (置信度: {final_confidence:.4f})")

    # --- 阶段五：保存结果 ---
    print("\n--- 阶段五：保存预测结果 ---")
    result_df = pd.DataFrame([
        {
            'filename': fname,
            'predicted_label': info['predicted_label'],
            'confidence': info['confidence'],
            'total_samples': info['total_samples']
        }
        for fname, info in file_predictions.items()
    ]).sort_values('filename').reset_index(drop=True)

    RESULTS_CSV = os.path.join(output_dir, '23_target_predictions_lightgbm.csv')
    result_df.to_csv(RESULTS_CSV, index=False)
    print(f"✅ 文件级预测结果已保存至: {RESULTS_CSV}")

    # --- 阶段六：可视化 ---
    print("\n--- 阶段六：可视化结果 ---")

    # 1. 预测类别分布饼图
    plt.figure(figsize=(8, 6))
    labels = result_df['predicted_label']
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors = sns.color_palette("husl", len(unique_labels))
    plt.pie(counts, labels=unique_labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('目标域文件预测类别分布 (LightGBM)', fontsize=14, weight='bold')
    plt.axis('equal')
    pie_path = os.path.join(output_dir, '23_prediction_distribution_pie.png')
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 预测分布饼图已保存至: {pie_path}")

    # 2. 置信度直方图
    plt.figure(figsize=(10, 6))
    confidences = result_df['confidence']
    plt.hist(confidences, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('文件预测置信度', fontsize=12)
    plt.ylabel('文件数量', fontsize=12)
    plt.title('目标域文件预测置信度分布 (LightGBM)', fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3)
    mean_conf = np.mean(confidences)
    median_conf = np.median(confidences)
    stats_text = f'均值: {mean_conf:.3f}\n中位数: {median_conf:.3f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    hist_path = os.path.join(output_dir, '23_prediction_confidence_hist.png')
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 置信度直方图已保存至: {hist_path}")

    print(f"\n🏆 任务三迁移诊断完成！")
    print(f"   - 模型: LightGBM (无 DANN)")
    print(f"   - 预测方式: 样本级预测 → 文件级投票")
    print(f"   - 结果保存路径: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()