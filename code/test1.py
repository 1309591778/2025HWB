# 22_task3_prediction_and_visualization.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from matplotlib import font_manager
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import scipy.io
from scipy.signal import hilbert


# 设置中文字体
def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")


# 创建输出目录
def create_output_dir():
    """创建输出目录"""
    output_dir = os.path.join('..', 'data', 'processed', 'task3_outputs_dann')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# --- 新增：定义梯度反转层 (Gradient Reversal Layer) ---
@tf.custom_gradient
def grad_reverse(x, lambda_val=1.0):
    """梯度反转函数"""
    y = tf.identity(x)

    def custom_grad(dy):
        return -dy * lambda_val, None

    return y, custom_grad


class GradReverse(tf.keras.layers.Layer):
    """梯度反转层 Keras 封装"""

    def __init__(self, lambda_val=1.0, **kwargs):
        super(GradReverse, self).__init__(**kwargs)
        self.lambda_val = tf.Variable(lambda_val, trainable=False, dtype=tf.float32)

    def call(self, x):
        return grad_reverse(x, self.lambda_val)

    def get_config(self):
        config = super(GradReverse, self).get_config()
        config.update({'lambda_val': float(self.lambda_val.numpy())})
        return config


# --- 新增结束 ---


# --- 新增：t-SNE 可视化 ---
def visualize_tsne(X_source, y_source, X_target, source_le, target_predictions_dict, output_dir, title_suffix=""):
    """使用 t-SNE 可视化源域和目标域数据"""
    print(f"  - 正在对特征进行 t-SNE 降维 ({title_suffix})...")
    try:
        # 合并特征和标签
        all_features = np.vstack([X_source, X_target])
        # 创建域标签
        domain_labels = np.array(['Source'] * len(X_source) + ['Target'] * len(X_target))

        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(all_features)

        plt.figure(figsize=(12, 10))

        # 绘制源域数据（按类别着色）
        unique_source_classes = np.unique(y_source)
        colors_classes = plt.cm.get_cmap('tab10', len(unique_source_classes))

        # 修复t-SNE可视化问题：扩展标签数组以匹配合并特征的长度
        extended_y_source = np.concatenate([y_source, np.zeros(len(X_target))])  # 扩展源域标签

        for i, cls_int in enumerate(unique_source_classes):
            cls_name = source_le.inverse_transform([cls_int])[0]
            # 修正：使用扩展的标签数组进行筛选
            source_mask = domain_labels == 'Source'
            cls_mask = extended_y_source == cls_int
            idx = source_mask & cls_mask
            plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1],
                        c=[colors_classes(i)], label=f'Source-{cls_name}', alpha=0.6, s=20)

        # 绘制目标域数据（按预测类别着色）
        target_filenames_in_order = list(target_predictions_dict.keys())
        y_target_pred_str = [target_predictions_dict[fname] for fname in target_filenames_in_order]

        # 处理目标域预测标签不在源域标签中的情况
        valid_target_mask = np.array([label in source_le.classes_ for label in y_target_pred_str])
        valid_target_filenames = np.array(target_filenames_in_order)[valid_target_mask]
        valid_y_target_pred_str = np.array(y_target_pred_str)[valid_target_mask]

        if len(valid_y_target_pred_str) > 0:
            try:
                y_target_pred_int_valid = source_le.transform(valid_y_target_pred_str)
                target_indices = np.where(domain_labels == 'Target')[0]
                # 修复维度不匹配问题
                valid_target_mask_full = np.isin(domain_labels, ['Target']) & np.isin(
                    domain_labels[domain_labels == 'Target'], ['Target'])

                # 正确的索引映射
                target_filename_map = {}
                for i, fname in enumerate(target_filenames):
                    if fname not in target_filename_map:
                        target_filename_map[fname] = []
                    target_filename_map[fname].append(i)

                for i, cls_int in enumerate(unique_source_classes):
                    cls_name = source_le.inverse_transform([cls_int])[0]
                    # 找到该类别的文件
                    files_for_class = [fname for fname, pred in target_predictions_dict.items() if
                                       pred['predicted_label'] == cls_name]

                    for fname in files_for_class:
                        if fname in target_filename_map:
                            file_indices = target_filename_map[fname]
                            # 在tsne_results中找到对应的目标域索引
                            target_start_idx = len(X_source)
                            tsne_file_indices = [target_start_idx + idx for idx in file_indices]
                            # 确保索引在范围内
                            tsne_file_indices = [idx for idx in tsne_file_indices if idx < len(tsne_results)]
                            if tsne_file_indices:
                                plt.scatter(tsne_results[tsne_file_indices, 0], tsne_results[tsne_file_indices, 1],
                                            c=[colors_classes(i)], label=f'Target-{cls_name}', alpha=0.5, s=20,
                                            marker='x')
            except ValueError as e:
                print(f"  ⚠️  标签转换错误: {e}。将使用统一颜色绘制目标域数据。")
                target_indices = np.where(domain_labels == 'Target')[0]
                plt.scatter(tsne_results[target_indices, 0], tsne_results[target_indices, 1],
                            c='red', label='Target (Uncertain Labels)', alpha=0.5, s=20, marker='x')
        else:
            print("  ⚠️  没有有效的目标域预测标签用于可视化。")
            target_indices = np.where(domain_labels == 'Target')[0]
            plt.scatter(tsne_results[target_indices, 0], tsne_results[target_indices, 1],
                        c='red', label='Target (No Valid Labels)', alpha=0.5, s=20, marker='x')

        plt.title(f't-SNE 可视化: 源域与目标域特征分布 {title_suffix}', fontsize=16, weight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'22_tsne_features_{title_suffix.replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ✅ t-SNE 图已保存至: {save_path}")
    except Exception as e:
        print(f"  - ⚠️ t-SNE 可视化失败: {e}")
        import traceback
        traceback.print_exc()


# --- 新增结束 ---


# 主程序
if __name__ == "__main__":
    set_chinese_font()
    output_dir = create_output_dir()

    print("🚀 任务三：预测与可视化 (基于已训练的DANN模型) 开始执行...")

    # --- 阶段一：加载源域模型与预处理器 ---
    print("\n--- 阶段一：加载源域模型与预处理器 ---")

    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    TASK2_OUTPUTS_DIR = os.path.join(PROCESSED_DIR, 'task2_outputs_final')

    SCALER_PATH = os.path.join(TASK2_OUTPUTS_DIR, 'final_scaler.joblib')
    LE_PATH = os.path.join(TASK2_OUTPUTS_DIR, 'final_label_encoder.joblib')

    try:
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LE_PATH)
        print("✅ 成功加载 StandardScaler 和 LabelEncoder。")
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件 {e.filename}。请确保任务二已成功运行并生成了输出文件。")
        exit(1)
    except Exception as e:
        print(f"❌ 加载预处理器时发生错误: {e}")
        exit(1)

    # --- 阶段二：加载目标域数据 ---
    print("\n--- 阶段二：加载目标域数据 ---")

    # 1. 加载目标域数据
    TARGET_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'target_features.csv')

    try:
        df_target_features = pd.read_csv(TARGET_FEATURES_PATH)
        print(f"✅ 成功加载目标域特征数据: {df_target_features.shape}")

        # 2. 分离文件名和特征
        target_filenames = df_target_features['source_file']

        # 修复：获取与源域训练时相同的特征列（排除label、rpm、filename等）
        SOURCE_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features_selected.csv')
        df_source_features = pd.read_csv(SOURCE_FEATURES_PATH)
        selected_feature_names = df_source_features.drop(columns=['label', 'rpm', 'filename']).columns.tolist()

        X_target_raw = df_target_features[selected_feature_names]  # 使用与源域相同的特征

        # 3. 标准化特征
        X_target_scaled = scaler.transform(X_target_raw)

        print(f"✅ 目标域数据预处理完成。")

    except FileNotFoundError as e:
        print(f"❌ 错误：找不到数据文件 {e.filename}。")
        exit(1)
    except Exception as e:
        print(f"❌ 处理数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- 阶段三：加载已训练的5个模型并进行预测 ---
    print("\n--- 阶段三：加载已训练模型并进行预测 ---")

    # 使用5个随机种子的模型
    random_seeds = [42, 123, 456, 789, 999]
    all_model_file_predictions = []  # 存储每个模型的文件级预测结果

    for seed_idx, seed in enumerate(random_seeds):
        print(f"\n--- 加载第 {seed_idx + 1} 个模型 (随机种子: {seed}) ---")

        try:
            # 加载已训练的模型
            model_save_path = os.path.join(output_dir, f'dann_model_seed_{seed}.h5')
            print(f"  - 正在加载模型: {model_save_path}")

            # 加载模型，包含自定义层
            loaded_model = tf.keras.models.load_model(model_save_path, custom_objects={'GradReverse': GradReverse})
            print("  - 模型加载成功。")

            # ✅ 关键修改：仅使用分类头进行预测，不依赖 domain adaptation
            print("  - 构建纯分类器（仅使用 class_output 层）...")
            classifier = Model(inputs=loaded_model.input, outputs=loaded_model.get_layer('class_output').output)

            # 执行预测（使用 training=False 避免 dropout 影响）
            print("  - 正在对目标域数据进行预测...")
            y_target_pred_proba = classifier.predict(X_target_scaled, batch_size=32)
            y_target_pred_int = np.argmax(y_target_pred_proba, axis=1)
            y_target_pred_labels = le.inverse_transform(y_target_pred_int)

            # 按文件名分组预测结果进行投票 - 每个模型单独进行
            target_results_df = pd.DataFrame({
                'filename': target_filenames,
                'predicted_label': y_target_pred_labels,
                'confidence': np.max(y_target_pred_proba, axis=1)
            })

            # 按文件名分组并投票
            file_predictions_for_this_model = {}
            for filename in target_filenames.unique():
                file_data = target_results_df[target_results_df['filename'] == filename]
                # 投票决定文件类别
                votes = file_data['predicted_label'].value_counts()
                predicted_class = votes.index[0]  # 得票最多的类别
                confidence = file_data['confidence'].mean()  # 平均置信度
                file_predictions_for_this_model[filename] = {'predicted_label': predicted_class,
                                                             'confidence': confidence}

            all_model_file_predictions.append(file_predictions_for_this_model)
            print(f"✅ 第 {seed_idx + 1} 个模型预测完成。")

        except Exception as e:
            print(f"❌ 在第 {seed_idx + 1} 个模型加载或预测阶段发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    # --- 阶段四：集成预测结果 ---
    print("\n--- 阶段四：集成预测结果 ---")

    if len(all_model_file_predictions) == 0:
        print("❌ 所有模型加载失败，无法进行集成预测。")
        exit(1)

    # 将5个模型的文件级预测结果进行最终投票
    final_file_predictions = {}
    for filename in target_filenames.unique():
        # 收集5个模型对这个文件的预测
        model_predictions = []
        confidence_scores = []

        for model_pred_dict in all_model_file_predictions:
            if filename in model_pred_dict:
                model_predictions.append(model_pred_dict[filename]['predicted_label'])
                confidence_scores.append(model_pred_dict[filename]['confidence'])

        # 投票决定最终类别
        votes = pd.Series(model_predictions).value_counts()
        final_predicted_class = votes.index[0]  # 得票最多的类别

        # 计算最终置信度
        vote_ratio = votes.iloc[0] / len(model_predictions)
        avg_confidence = np.mean(confidence_scores)
        final_confidence = vote_ratio * avg_confidence

        final_file_predictions[filename] = {
            'predicted_label': final_predicted_class,
            'confidence': final_confidence
        }

    print("✅ 集成预测完成。")

    # --- 阶段五：迁移结果可视化展示与分析 ---
    print("\n--- 阶段五：迁移结果可视化展示与分析 ---")

    # 1. 文件级预测标签分布
    file_pred_labels = [v['predicted_label'] for v in final_file_predictions.values()]
    unique_labels, counts = np.unique(file_pred_labels, return_counts=True)
    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("husl", len(unique_labels))
    plt.pie(counts, labels=unique_labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('目标域文件预测结果类别分布 (DANN 集成)', fontsize=14, weight='bold')
    plt.axis('equal')
    save_path_pie = os.path.join(output_dir, '22_target_prediction_distribution_dann_ensemble.png')
    plt.savefig(save_path_pie, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 文件级预测标签分布饼图已保存至: {save_path_pie}")

    # 2. 文件级预测置信度分析
    file_confidences = [v['confidence'] for v in final_file_predictions.values()]
    plt.figure(figsize=(10, 6))
    plt.hist(file_confidences, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('文件预测置信度', fontsize=12)
    plt.ylabel('文件数量', fontsize=12)
    plt.title('目标域文件预测置信度分布 (DANN 集成)', fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3)
    mean_conf = np.mean(file_confidences)
    median_conf = np.median(file_confidences)
    stats_text = f'均值: {mean_conf:.3f}\n中位数: {median_conf:.3f}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    save_path_hist = os.path.join(output_dir, '22_prediction_confidence_histogram_dann_ensemble.png')
    plt.savefig(save_path_hist, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 文件级预测置信度直方图已保存至: {save_path_hist}")
    print(f"📊 文件级预测置信度统计 - 均值: {mean_conf:.4f}, 中位数: {median_conf:.4f}")

    # 3. 保存预测结果
    file_results_df = pd.DataFrame({
        'filename': list(final_file_predictions.keys()),
        'predicted_label': [v['predicted_label'] for v in final_file_predictions.values()],
        'confidence': [v['confidence'] for v in final_file_predictions.values()]
    })
    file_results_df = file_results_df.sort_values(by='filename').reset_index(drop=True)
    RESULTS_CSV_PATH = os.path.join(output_dir, '22_target_domain_predictions_dann_ensemble.csv')
    file_results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"✅ 目标域文件预测结果已保存至: {RESULTS_CSV_PATH}")

    # 4. 加载源域数据用于t-SNE可视化
    print("\n--- 阶段六：t-SNE可视化 ---")
    SOURCE_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features_selected.csv')
    try:
        df_source_features = pd.read_csv(SOURCE_FEATURES_PATH)
        selected_feature_names = df_source_features.drop(columns=['label', 'rpm', 'filename']).columns.tolist()
        X_source_raw = df_source_features[selected_feature_names]
        y_source_str = df_source_features['label']
        y_source = le.transform(y_source_str)
        X_source_scaled = scaler.transform(X_source_raw)

        # t-SNE 可视化 (迁移前后对比)
        visualize_tsne(X_source_scaled, y_source, X_target_scaled, le,
                       {fname: v for fname, v in final_file_predictions.items()},
                       output_dir, title_suffix="迁移后_集成")
    except Exception as e:
        print(f"❌ t-SNE可视化准备失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n🏆 任务三预测与可视化完成！")
    print(f"   - 使用的基础模型: 已训练的5个DANN模型（仅使用分类头）")
    print(f"   - 预测方式: 集成投票 (按文件分组)")
    print(f"   - 预测结果已保存在: {RESULTS_CSV_PATH}")
    print(f"   - 可视化图表已保存在: {os.path.abspath(output_dir)}")