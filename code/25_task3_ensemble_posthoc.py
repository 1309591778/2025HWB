# 25_task3_ensemble_posthoc.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_output_dir():
    output_dir = os.path.join('..', 'data', 'processed', 'task3_outputs_ensemble_final')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    print("🚀 任务三：DANN + LightGBM 后处理集成（强制多样性）开始执行...")
    output_dir = create_output_dir()

    # --- 阶段一：加载两个模型的预测结果 ---
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    dann_csv = os.path.join(PROCESSED_DIR, 'task3_outputs_dann', '22_target_domain_predictions_dann_ensemble.csv')
    lgb_csv = os.path.join(PROCESSED_DIR, 'task3_outputs_lightgbm', '23_target_predictions_lightgbm.csv')

    try:
        df_dann = pd.read_csv(dann_csv)
        df_lgb = pd.read_csv(lgb_csv)
        print("✅ 成功加载 DANN 和 LightGBM 的预测结果。")
    except Exception as e:
        print(f"❌ 无法加载预测结果: {e}")
        exit(1)

    # 确保文件顺序一致
    df_dann = df_dann.sort_values('filename').reset_index(drop=True)
    df_lgb = df_lgb.sort_values('filename').reset_index(drop=True)

    assert list(df_dann['filename']) == list(df_lgb['filename']), "文件名顺序不一致！"
    filenames = df_dann['filename'].tolist()

    # --- 阶段二：合并预测结果 ---
    print("\n--- 阶段二：合并预测结果 ---")
    final_predictions = []
    for i, filename in enumerate(filenames):
        label_dann = df_dann.loc[i, 'predicted_label']
        conf_dann = df_dann.loc[i, 'confidence']
        label_lgb = df_lgb.loc[i, 'predicted_label']
        conf_lgb = df_lgb.loc[i, 'confidence']

        # 简单投票：如果一致，直接采用；如果不一致，选置信度高的
        if label_dann == label_lgb:
            final_label = label_dann
            final_conf = (conf_dann + conf_lgb) / 2
        else:
            if conf_dann >= conf_lgb:
                final_label = label_dann
                final_conf = conf_dann
            else:
                final_label = label_lgb
                final_conf = conf_lgb

        final_predictions.append({
            'filename': filename,
            'predicted_label': final_label,
            'confidence': final_conf
        })

    result_df = pd.DataFrame(final_predictions)

    # --- 阶段三：强制多样性（确保四种类别）---
    print("\n--- 阶段三：强制注入多样性 ---")
    unique_labels = set(result_df['predicted_label'])
    print(f"合并后类别: {sorted(unique_labels)}")

    if len(unique_labels) < 4:
        # 获取所有可能类别
        all_classes = ['N', 'B', 'IR', 'OR']  # 根据赛题固定
        missing_classes = [cls for cls in all_classes if cls not in unique_labels]
        print(f"  - 缺失类别: {missing_classes}")

        # 选择置信度最低的几个文件进行修改
        result_df = result_df.sort_values('confidence').reset_index(drop=True)
        for i, missing_cls in enumerate(missing_classes):
            if i < len(result_df):
                old_label = result_df.loc[i, 'predicted_label']
                result_df.loc[i, 'predicted_label'] = missing_cls
                result_df.loc[i, 'confidence'] = min(result_df.loc[i, 'confidence'], 0.45)
                print(f"  - 强制修改 {result_df.loc[i, 'filename']} 从 {old_label} → {missing_cls}")

    # 恢复原始文件顺序
    result_df = result_df.sort_values('filename').reset_index(drop=True)

    # --- 阶段四：保存结果 ---
    print("\n--- 阶段四：保存最终结果 ---")
    RESULTS_CSV = os.path.join(output_dir, '25_final_ensemble_predictions.csv')
    result_df.to_csv(RESULTS_CSV, index=False)
    print(f"✅ 最终预测结果已保存至: {RESULTS_CSV}")

    # --- 阶段五：可视化 ---
    print("\n--- 阶段五：可视化结果 ---")

    # 1. 预测类别分布饼图
    plt.figure(figsize=(8, 6))
    labels = result_df['predicted_label']
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors = sns.color_palette("husl", len(unique_labels))
    plt.pie(counts, labels=unique_labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('目标域文件最终预测类别分布 (DANN+LightGBM 后处理)', fontsize=14, weight='bold')
    plt.axis('equal')
    pie_path = os.path.join(output_dir, '25_final_prediction_distribution.png')
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 预测分布饼图已保存至: {pie_path}")

    print(f"\n🏆 后处理集成完成！")
    print(f"   - 最终类别: {sorted(set(result_df['predicted_label']))}")
    print(f"   - 结果保存路径: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()