# 12_task2_additional_visualizations.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import joblib
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import shap
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, LSTM, \
    GlobalAveragePooling1D, Dense, Dropout, multiply, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import lightgbm as lgb
from sklearn import svm

# ==================== 添加模型定义函数 ====================
def categorical_focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for addressing class imbalance in categorical classification
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        weight = alpha * y_true * tf.keras.backend.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        loss = tf.keras.backend.sum(loss, axis=1)
        return loss
    return categorical_focal_loss_fixed


def create_cnn_lstm_model(input_shape, num_classes):
    """创建简单的CNN+LSTM模型"""
    inputs = Input(shape=input_shape)

    # CNN特征提取层
    x = Conv1D(filters=64, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    # LSTM时序建模层
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32, return_sequences=False)(x)

    # 全连接分类层
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # 注意：在加载权重时，通常不需要编译模型。
    # 但如果后续需要（例如微调），可以取消下面的注释并确保 categorical_focal_loss 可用。
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #     loss=categorical_focal_loss(gamma=2., alpha=0.5),
    #     metrics=['accuracy']
    # )
    return model
# ==================== 添加模型定义函数结束 ====================


# 设置中文字体
def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")


# 创建输出目录
def create_output_dir():
    """创建可视化输出目录"""
    output_dir = os.path.join('..', 'data', 'processed', 'task2_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# 1. SHAP特征重要性图（Top 10特征）- XGBoost (修复版)
def plot_shap_feature_importance(X, model, feature_names, output_dir):
    """绘制SHAP特征重要性图"""
    print("  - 正在生成SHAP特征重要性图...")

    try:
        # 确保X是numpy数组且形状正确
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)

        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)

        # 为了提高效率，只使用部分数据计算SHAP值
        sample_size = min(500, X_array.shape[0])  # 减少样本数避免内存问题
        sample_indices = np.random.choice(X_array.shape[0], sample_size, replace=False)
        X_sample = X_array[sample_indices]

        # 确保特征名称是列表
        if isinstance(feature_names, pd.Index):
            feature_names_list = feature_names.tolist()
        else:
            feature_names_list = list(feature_names)

        # 确保X_sample是二维数组且列数与特征名称匹配
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(-1, 1)

        # 如果特征数量不匹配，截取或补齐
        if X_sample.shape[1] != len(feature_names_list):
            min_features = min(X_sample.shape[1], len(feature_names_list))
            X_sample = X_sample[:, :min_features]
            feature_names_list = feature_names_list[:min_features]
            print(f"  - ⚠️ 特征数量不匹配，已调整为 {min_features} 个特征")

        # 创建DataFrame确保列名正确
        X_sample_df = pd.DataFrame(X_sample, columns=feature_names_list)

        # 计算SHAP值
        shap_values = explainer.shap_values(X_sample_df)

        # 如果是多分类，shap_values是一个列表
        if isinstance(shap_values, list):
            # 对于多分类，我们计算每个类别的平均绝对SHAP值
            shap_importance = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(0)

        # 确保shap_importance是一维数组且长度与特征名称匹配
        if hasattr(shap_importance, 'ndim') and shap_importance.ndim > 1:
            shap_importance = shap_importance.flatten()

        if len(shap_importance) != len(feature_names_list):
            min_len = min(len(shap_importance), len(feature_names_list))
            shap_importance = shap_importance[:min_len]
            feature_names_list = feature_names_list[:min_len]

        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names_list,
            'importance': np.abs(shap_importance)  # 确保是绝对值
        }).sort_values('importance', ascending=False)

        # 绘制Top 10特征重要性
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(10)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title('Top 10 SHAP特征重要性 (XGBoost)', fontsize=16, weight='bold')
        plt.xlabel('平均SHAP值', fontsize=12)
        plt.ylabel('特征名称', fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(output_dir, '12_1_shap_feature_importance_xgboost.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ✅ SHAP特征重要性图已保存至: {save_path}")

    except Exception as e:
        print(f"  - ⚠️ 生成SHAP特征重要性图时出错: {e}")
        import traceback
        traceback.print_exc()


# 2. 类别分布饼图 (作为问题背景说明)
def plot_class_distribution(y_true, class_names, output_dir):
    """绘制类别分布饼图"""
    print("  - 正在生成类别分布饼图...")

    try:
        # 统计各类别数量
        unique, counts = np.unique(y_true, return_counts=True)
        class_counts = dict(zip(unique, counts))
        class_labels = [class_names[i] for i in unique]
        class_values = [class_counts[i] for i in unique]

        # 绘制饼图
        plt.figure(figsize=(10, 8))
        colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'orange']
        plt.pie(class_values, labels=class_labels, colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title('源域数据集类别分布\n(用于揭示数据不平衡挑战)', fontsize=16, weight='bold')
        plt.axis('equal')

        save_path = os.path.join(output_dir, '12_4_class_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ✅ 类别分布饼图已保存至: {save_path}")

    except Exception as e:
        print(f"  - ⚠️ 生成类别分布饼图时出错: {e}")


# 3. 混淆矩阵热力图（综合）
def plot_confusion_matrix_heatmap(y_true, y_pred_models, class_names, output_dir):
    """绘制综合混淆矩阵热力图"""
    print("  - 正在生成综合混淆矩阵热力图...")

    try:
        # 选择最佳模型（这里选择第一个模型）进行详细分析
        best_model_name = list(y_pred_models.keys())[0]
        y_pred_best = y_pred_models[best_model_name]

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred_best)

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{best_model_name} 模型混淆矩阵', fontsize=16, weight='bold')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')

        save_path = os.path.join(output_dir, '12_5_confusion_matrix_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ✅ 综合混淆矩阵热力图已保存至: {save_path}")

    except Exception as e:
        print(f"  - ⚠️ 生成综合混淆矩阵热力图时出错: {e}")


# 4. 业务价值导向评价图（安全性-经济性权衡）
def plot_safety_economy_tradeoff(models_reports, output_dir):
    """绘制安全性和经济性权衡图"""
    print("  - 正在生成安全性-经济性权衡图...")

    try:
        models_names = list(models_reports.keys())
        safety_scores = []  # IR类召回率（安全性）
        economy_scores = []  # B类精确率（经济性）

        for model_name, report in models_reports.items():
            ir_recall = report.get('IR', {}).get('recall', 0)
            b_precision = report.get('B', {}).get('precision', 0)
            safety_scores.append(ir_recall)
            economy_scores.append(b_precision)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(economy_scores, safety_scores, s=150, alpha=0.7, c=range(len(models_names)),
                              cmap='viridis')

        # 添加模型名称标签
        for i, model_name in enumerate(models_names):
            plt.annotate(model_name, (economy_scores[i], safety_scores[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')

        plt.xlabel('经济性指标 (B类精确率)\n↑ 避免误判，减少不必要停机', fontsize=12)
        plt.ylabel('安全性指标 (IR类召回率)\n↑ 避免漏检，保障设备安全', fontsize=12)
        plt.title('模型安全性-经济性权衡分析\n(左上角为理想区域)', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3)

        # 添加象限参考线 (x=0.98, y=0.98)，代表理想目标
        plt.axhline(y=0.98, color='r', linestyle='--', alpha=0.5, linewidth=1)
        plt.axvline(x=0.98, color='r', linestyle='--', alpha=0.5, linewidth=1)

        # 将坐标轴范围调整到更精细的区间，以突出微小差异
        plt.xlim(0.96, 1.01)
        plt.ylim(0.96, 1.01)

        save_path = os.path.join(output_dir, '12_6_safety_economy_tradeoff.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ✅ 安全性-经济性权衡图已保存至: {save_path}")

    except Exception as e:
        print(f"  - ⚠️ 生成安全性-经济性权衡图时出错: {e}")


# 5. 关键风险指标雷达图
def plot_key_risk_indicators(models_reports, output_dir):
    """绘制关键风险指标雷达图"""
    print("  - 正在生成关键风险指标雷达图...")

    try:
        models_names = list(models_reports.keys())

        # 提取关键指标
        indicators = ['IR召回率', 'B精确率', 'N召回率', 'OR精确率', '总体准确率']
        data = {}

        for model_name, report in models_reports.items():
            data[model_name] = [
                report.get('IR', {}).get('recall', 0),
                report.get('B', {}).get('precision', 0),
                report.get('N', {}).get('recall', 0),
                report.get('OR', {}).get('precision', 0),
                report.get('accuracy', 0)
            ]

        # 绘制雷达图
        labels = np.array(indicators)
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, values) in enumerate(data.items()):
            values += values[:1]  # 闭合图形
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('关键风险指标雷达图\n(越靠近边缘越好)', size=16, weight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(output_dir, '12_7_key_risk_indicators.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ✅ 关键风险指标雷达图已保存至: {save_path}")

    except Exception as e:
        print(f"  - ⚠️ 生成关键风险指标雷达图时出错: {e}")


# 6. 综合性能评分卡
def plot_comprehensive_scorecard(models_reports, output_dir):
    """绘制综合性能评分卡"""
    print("  - 正在生成综合性能评分卡...")

    try:
        models_names = list(models_reports.keys())

        # 定义评分维度和权重
        scoring_criteria = {
            'accuracy': 0.3,  # 总体准确率 30%
            'IR_recall': 0.25,  # IR召回率 25%
            'B_precision': 0.2,  # B精确率 20%
            'N_recall': 0.15,  # N召回率 15%
            'macro_f1': 0.1  # 宏平均F1 10%
        }

        # 计算各模型综合得分
        scores = {}
        for model_name, report in models_reports.items():
            score = 0
            score += report.get('accuracy', 0) * scoring_criteria['accuracy']
            score += report.get('IR', {}).get('recall', 0) * scoring_criteria['IR_recall']
            score += report.get('B', {}).get('precision', 0) * scoring_criteria['B_precision']
            score += report.get('N', {}).get('recall', 0) * scoring_criteria['N_recall']
            # 使用 'macro avg' 或直接从报告中获取宏F1
            macro_f1 = report.get('macro avg', {}).get('f1-score', 0)
            if macro_f1 == 0:
                # 如果 'macro avg' 键不存在，尝试从 'weighted avg' 或其他方式获取，或者用0代替
                macro_f1 = 0
            score += macro_f1 * scoring_criteria['macro_f1']
            scores[model_name] = score * 100  # 转换为百分制

        # 绘制评分卡
        plt.figure(figsize=(16, 6))

        # 左侧：综合得分柱状图
        models_list = list(scores.keys())
        scores_list = list(scores.values())

        bars = plt.bar(models_list, scores_list, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
        plt.ylabel('综合得分 (满分100)', fontsize=12)
        plt.title('模型综合性能评分卡\n(加权多维度评估)', fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, score in zip(bars, scores_list):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{score:.1f}', ha='center', va='bottom', fontsize=12, weight='bold')

        plt.ylim(0, 100)
        plt.xticks(rotation=45)

        # 右侧：最佳模型详细性能雷达图
        # 找到综合得分最高的模型
        best_model = max(scores, key=scores.get)
        best_report = models_reports[best_model]

        # 定义一个函数来安全地获取数值
        def get_safe_value(report, key, default=0):
            try:
                if isinstance(report, dict):
                    return report.get(key, default)
                else:
                    return default
            except:
                return default

        # 使用安全函数获取雷达图数据
        radar_values = [
            get_safe_value(best_report, 'accuracy'),
            get_safe_value(best_report, 'IR', {}).get('recall', 0),
            get_safe_value(best_report, 'B', {}).get('precision', 0),
            get_safe_value(best_report, 'N', {}).get('recall', 0),
            get_safe_value(best_report, 'macro avg', {}).get('f1-score', 0)
        ]

        # 确保数组长度正确
        if len(radar_values) != 5:
            print(f"⚠️ 警告: {best_model} 的雷达图数据长度异常，已填充默认值")
            radar_values = [0] * 5  # 或者根据实际情况处理

        # 绘制雷达图
        ax2 = plt.subplot(1, 2, 2, projection='polar')
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
        angles += angles[:1]
        radar_values += radar_values[:1]

        ax2.plot(angles, radar_values, 'o-', linewidth=2, label=best_model, color='red')
        ax2.fill(angles, radar_values, alpha=0.25, color='red')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(['accuracy', 'IR_recall', 'B_precision', 'N_recall', 'macro_f1'], fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_title(f'{best_model} 详细性能雷达图', size=14, weight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, '12_10_comprehensive_scorecard.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ✅ 综合性能评分卡已保存至: {save_path}")

    except Exception as e:
        print(f"  - ⚠️ 生成综合性能评分卡时出错: {e}")


# 7. 细化混淆矩阵（突出 IR/N）
def plot_detailed_confusion_matrix(y_true, y_pred_models, class_names, output_dir):
    """绘制细化的混淆矩阵，突出显示 IR 和 N 类"""
    print("  - 正在生成细化混淆矩阵图...")

    try:
        # 假设 IR 是 'IR'，N 是 'N'
        ir_index = list(class_names).index('IR')
        n_index = list(class_names).index('N')
        target_indices = [ir_index, n_index]
        target_labels = ['IR', 'N']

        for model_name, y_pred in y_pred_models.items():
            # 计算完整混淆矩阵
            cm_full = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

            # 提取 IR 和 N 的子矩阵
            cm_sub = cm_full[np.ix_(target_indices, target_indices)]

            # 绘制 IR/N 子矩阵
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_sub, annot=True, fmt='d', cmap='Reds',  # 使用红色系突出
                        xticklabels=target_labels, yticklabels=target_labels,
                        cbar_kws={"shrink": .8})
            plt.title(f'{model_name} 模型 IR/N 类混淆矩阵', fontsize=14, weight='bold')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.tight_layout()

            save_path = os.path.join(output_dir, f'13_detailed_cm_{model_name}_IR_N.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  - ✅ {model_name} 细化混淆矩阵 (IR/N) 已保存至: {save_path}")

    except Exception as e:
        print(f"  - ⚠️ 生成细化混淆矩阵图时出错: {e}")
        import traceback
        traceback.print_exc()


# --- 修改：更新函数签名和内部逻辑 ---
# 8. 关键类性能稳定性图（方案B：简化版）
# def plot_key_class_performance_summary(models_reports, output_dir): # <--- 旧签名
def plot_key_class_performance_summary(y_pred_models_dict, class_names, y_true_input, output_dir): # <--- 新签名
    """绘制关键类性能汇总图（简化版，非跨折稳定性）"""
    print("  - 正在生成关键类性能汇总图...")
    try:
        # models_names = list(models_reports.keys()) # <--- 旧逻辑
        models_names = list(y_pred_models_dict.keys()) # <--- 新逻辑
        # ir_recalls = [models_reports[m].get('IR', {}).get('recall', 0) for m in models_names] # <--- 旧逻辑
        # n_recalls = [models_reports[m].get('N', {}).get('recall', 0) for m in models_names] # <--- 旧逻辑
        # n_f1s = [models_reports[m].get('N', {}).get('f1-score', 0) for m in models_names] # <--- 旧逻辑

        # --- 新增：根据 y_pred 重新计算指标 ---
        ir_recalls = []
        n_recalls = []
        n_f1s = []
        for model_name in models_names:
            y_pred = y_pred_models_dict[model_name]
            # 使用 sklearn.metrics.classification_report 临时计算
            tmp_report = classification_report(y_true_input, y_pred, target_names=class_names, output_dict=True)
            ir_recalls.append(tmp_report.get('IR', {}).get('recall', 0))
            n_recalls.append(tmp_report.get('N', {}).get('recall', 0))
            n_f1s.append(tmp_report.get('N', {}).get('f1-score', 0))
        # --- 新增结束 ---

        x = np.arange(len(models_names))  # 标签位置
        width = 0.25  # 柱状图的宽度

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width, ir_recalls, width, label='IR 召回率', color='skyblue')
        rects2 = ax.bar(x, n_recalls, width, label='N 召回率', color='lightcoral')
        rects3 = ax.bar(x + width, n_f1s, width, label='N F1分数', color='lightgreen')

        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        ax.set_ylabel('分数')
        ax.set_title('各模型关键类 (IR, N) 性能汇总')
        ax.set_xticks(x)
        ax.set_xticklabels(models_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        save_path = os.path.join(output_dir, '14_key_class_performance_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ✅ 关键类性能汇总图已保存至: {save_path}")

    except Exception as e:
        print(f"  - ⚠️ 生成关键类性能汇总图时出错: {e}")
        import traceback
        traceback.print_exc()
# --- 修改结束 ---


# --- 修改：生成所有模型的混淆矩阵和精度对比图 ---
# def plot_model_comparison_charts(models_reports, output_dir): # <--- 旧签名
def plot_model_comparison_charts(y_pred_models_dict, class_names, y_true_input, output_dir): # <--- 新签名
    """生成所有模型的性能对比图表"""
    print("  - 正在生成所有模型性能对比图表...")
    try:
        # 1. 混淆矩阵对比 (子图)
        # 假设最多6个模型，用2x3布局
        num_models = len(y_pred_models_dict)
        rows = (num_models + 2) // 3 # 向上取整计算行数
        cols = 3 if num_models > 1 else 1
        if num_models <= 2:
            rows = 1
            cols = num_models

        fig_cm, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        # 处理只有一个子图的情况
        if num_models == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.ravel()

        for idx, (model_name, y_pred) in enumerate(y_pred_models_dict.items()): # <--- 修改：遍历 y_pred_models_dict
            if idx < len(axes):
                # --- 修改：使用传入的 y_true_input 和 y_pred ---
                cm = confusion_matrix(y_true_input, y_pred) # <--- 修改这里 ---
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            # --- 修改：使用传入的 class_names ---
                            xticklabels=class_names, yticklabels=class_names, # <--- 修改这里 ---
                            ax=axes[idx], cbar_kws={"shrink": .8}) # 添加颜色条缩小
                axes[idx].set_title(f'{model_name} 混淆矩阵', fontsize=14)
                axes[idx].set_xlabel('Predicted Label')
                axes[idx].set_ylabel('True Label')

        # 隐藏多余的子图
        for j in range(len(y_pred_models_dict), len(axes)): # <--- 修改：使用 len(y_pred_models_dict) ---
            axes[j].set_visible(False)

        plt.tight_layout()
        save_path_cm_all = os.path.join(output_dir, '12_11_all_models_confusion_matrices.png')
        plt.savefig(save_path_cm_all, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ✅ 所有模型混淆矩阵对比图已保存至: {save_path_cm_all}")

        # 2. 精度/指标对比 (条形图)
        comparison_data = []
        # --- 修改：遍历 y_pred_models_dict 并计算指标 ---
        for model_name, y_pred in y_pred_models_dict.items(): # <--- 修改这里 ---
            # --- 修改：使用传入的 y_true_input 计算准确率 ---
            accuracy = accuracy_score(y_true_input, y_pred) # <--- 修改这里 ---
            # --- 修改：使用传入的 y_true_input 和 class_names 计算报告 ---
            report_dict = classification_report(y_true_input, y_pred, target_names=class_names, output_dict=True) # <--- 修改这里 ---
            macro_f1 = report_dict.get('macro avg', {}).get('f1-score', 0)
            weighted_f1 = report_dict.get('weighted avg', {}).get('f1-score', 0)
            ir_recall = report_dict.get('IR', {}).get('recall', 0)
            b_precision = report_dict.get('B', {}).get('precision', 0)
            n_recall = report_dict.get('N', {}).get('recall', 0) # 假设 N 类存在

            comparison_data.append({
                'Model': model_name,
                'Accuracy': accuracy, # <--- 使用计算出的准确率 ---
                'Macro F1': macro_f1,
                'Weighted F1': weighted_f1,
                'IR Recall': ir_recall,
                'B Precision': b_precision,
                'N Recall': n_recall
            })
        # --- 修改结束 ---

        df_comparison = pd.DataFrame(comparison_data)
        # melted_df = df_comparison.melt(id_vars=['Model'], var_name='Metric', value_name='Score')

        fig_metrics, ax = plt.subplots(figsize=(12, 8))

        # 使用 Seaborn 更简洁地绘制
        df_melted = df_comparison.melt(id_vars=['Model'], value_vars=['Accuracy', 'Macro F1', 'Weighted F1', 'IR Recall', 'B Precision', 'N Recall'],
                                     var_name='Metric', value_name='Score')
        sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, ax=ax)
        ax.set_title('模型性能指标对比 (5折交叉验证)', fontsize=16)
        ax.set_ylabel('Score')
        ax.set_xlabel('Model')
        ax.legend(title='Metric')
        plt.xticks(rotation=45)
        plt.tight_layout()

        save_path_metrics = os.path.join(output_dir, '12_12_model_performance_comparison.png')
        plt.savefig(save_path_metrics, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ✅ 模型性能指标对比图已保存至: {save_path_metrics}")

    except Exception as e:
        print(f"  - ⚠️ 生成性能对比图表时出错: {e}")
        import traceback
        traceback.print_exc()
# --- 修改结束 ---


# 主程序
if __name__ == "__main__":
    set_chinese_font()
    output_dir = create_output_dir()

    print("🚀 开始生成任务二额外可视化图表...")

    # 加载数据
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features_selected.csv')

    try:
        # 加载特征数据
        df_features = pd.read_csv(FEATURES_PATH)
        X_raw = df_features.drop(columns=['label', 'rpm', 'filename'])
        y_str = df_features['label']
        le = LabelEncoder()
        y = le.fit_transform(y_str)

        print(f"成功加载数据: {len(X_raw)} 个样本")

        # 加载训练好的模型
        XGB_MODEL_PATH = os.path.join(PROCESSED_DIR, 'task2_outputs_final', 'final_xgb_model.joblib')
        RF_MODEL_PATH = os.path.join(PROCESSED_DIR, 'task2_outputs_final', 'final_rf_model.joblib')
        SVM_MODEL_PATH = os.path.join(PROCESSED_DIR, 'task2_outputs_final', 'final_svm_model.joblib') # <--- 新增 ---
        LGB_MODEL_PATH = os.path.join(PROCESSED_DIR, 'task2_outputs_final', 'final_lgb_model.txt') # <--- 新增 ---
        # --- 修改点1: 更正变量名 ---
        CNN_LSTM_WEIGHTS_PATH = os.path.join(PROCESSED_DIR, 'task2_outputs_final', 'final_cnn_lstm_model.weights.h5') # <--- 修改这里 ---
        SCALER_PATH = os.path.join(PROCESSED_DIR, 'task2_outputs_final', 'final_scaler.joblib')

        models = {}
        if os.path.exists(XGB_MODEL_PATH):
            models['XGBoost'] = joblib.load(XGB_MODEL_PATH)
            print("成功加载XGBoost模型")
        else:
            print("未找到XGBoost模型")

        if os.path.exists(RF_MODEL_PATH):
            models['RandomForest'] = joblib.load(RF_MODEL_PATH)
            print("成功加载随机森林模型")
        else:
            print("未找到随机森林模型")

        # --- 新增：加载 SVM 模型 ---
        if os.path.exists(SVM_MODEL_PATH):
            models['SVM'] = joblib.load(SVM_MODEL_PATH)
            print("成功加载SVM模型")
        else:
            print("未找到SVM模型")
        # --- 新增结束 ---

        # --- 新增：加载 LightGBM 模型 ---
        if os.path.exists(LGB_MODEL_PATH):
            # LightGBM Booster 对象需要使用 lgb.Booster 加载
            models['LightGBM'] = lgb.Booster(model_file=LGB_MODEL_PATH)
            print("成功加载LightGBM模型")
        else:
            print("未找到LightGBM模型")
        # --- 新增结束 ---

        # --- 修改点2: 移除旧的加载逻辑，添加新的加载权重逻辑 ---
        # if os.path.exists(CNN_LSTM_MODEL_PATH):
        #     models['CNN-LSTM'] = tf.keras.models.load_model(CNN_LSTM_MODEL_PATH)  # 新增加载CNN-LSTM
        #     print("成功加载CNN-LSTM模型")
        # else:
        #     print("未找到CNN-LSTM模型")

        # --- 新增：加载 CNN-LSTM 模型权重 ---
        if os.path.exists(CNN_LSTM_WEIGHTS_PATH): # <--- 修改这里 ---
            # 1. 获取模型输入形状和类别数
            input_shape = (X_raw.shape[1], 1)  # (特征数, 1)
            num_classes = len(np.unique(y))    # 类别数

            # 2. 重新创建模型架构 (使用上面定义的函数)
            reconstructed_cnn_lstm_model = create_cnn_lstm_model(input_shape, num_classes) # <--- 修改这里 ---

            # 3. 加载权重到重建的模型中
            reconstructed_cnn_lstm_model.load_weights(CNN_LSTM_WEIGHTS_PATH) # <--- 修改这里 ---
            models['CNN-LSTM'] = reconstructed_cnn_lstm_model # <--- 修改模型字典中的键名 ---
            print("成功加载CNN-LSTM模型 (通过权重)") # <--- 修改这里 ---
        else:
            print("未找到CNN-LSTM模型权重文件") # <--- 修改这里 ---
        # --- 新增结束 ---

        # 1. SHAP特征重要性图 (XGBoost) - 修复版
        if 'XGBoost' in models:
            plot_shap_feature_importance(X_raw, models['XGBoost'], X_raw.columns, output_dir)

        # 2. 模型性能评估
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            X_scaled = scaler.transform(X_raw)

            # 获取预测结果和真实标签
            y_true = y # <--- 定义 y_true 供后续使用 ---
            global y_true_global # <--- 定义全局变量供函数使用 ---
            y_true_global = y_true
            global le_global # <--- 定义全局变量供函数使用 ---
            le_global = le

            # 计算各模型准确率和分类报告
            accuracies = {}
            reports = {} # 保存 classification_report 字典
            y_pred_models = {}  # 保存各模型预测结果数组 <--- 关键修改 ---

            for model_name, model in models.items():
                try:
                    if model_name == 'XGBoost':
                        y_pred = model.predict(X_scaled)
                    elif model_name == 'RandomForest':
                        y_pred = model.predict(X_scaled)
                    # --- 新增：SVM 模型预测逻辑 ---
                    elif model_name == 'SVM':
                        y_pred = model.predict(X_scaled)
                    # --- 新增结束 ---
                    # --- 新增：LightGBM 模型预测逻辑 ---
                    elif model_name == 'LightGBM':
                        # LightGBM Booster 需要使用 predict
                        y_pred_prob = model.predict(X_scaled)
                        y_pred = np.argmax(y_pred_prob, axis=1)
                    # --- 新增结束 ---
                    # --- 修改：修正 CNN-LSTM 模型的预测逻辑 ---
                    # elif model_name == 'CNN-LSTM': # <--- 旧的错误逻辑 ---
                    #     # CNN-LSTM 模型的输入是2D张量 (samples, features)
                    #     # 输出是 [class_output_probabilities, domain_output_probabilities]
                    #     X_scaled_cnn = np.expand_dims(X_scaled, axis=2) # <--- 添加这行 ---
                    #     y_pred_prob, _ = model.predict(X_scaled_cnn) # <--- 修改这里 ---
                    #     y_pred = np.argmax(y_pred_prob, axis=1)   # <--- 修改这里 ---
                    elif model_name == 'CNN-LSTM':  # <--- 新的正确逻辑 ---
                        # 1. 准备输入数据 (3D 张量)
                        X_scaled_cnn = np.expand_dims(X_scaled, axis=2)
                        # 2. 模型预测 (对于单输出模型，predict 返回单个数组)
                        y_pred_prob = model.predict(X_scaled_cnn)  # <--- 修改这里 ---
                        # 3. 获取预测类别
                        y_pred = np.argmax(y_pred_prob, axis=1)  # <--- 修改这里 ---
                    # --- 修改结束 ---
                    else:
                        continue

                    accuracy = accuracy_score(y_true, y_pred)
                    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)

                    accuracies[model_name] = accuracy
                    reports[model_name] = report
                    y_pred_models[model_name] = y_pred  # 保存预测结果数组

                    print(f"  - {model_name} 准确率: {accuracy:.4f}")
                except Exception as e:
                    print(f"  - ⚠️ 计算 {model_name} 性能时出错: {e}")
                    import traceback

                    traceback.print_exc()  # 打印完整错误堆栈，便于调试

            # 生成核心诊断结果评价可视化图表
            if reports and y_pred_models: # <--- 修改：确保两个字典都不为空 ---
                # 生成所有评价图表
                plot_safety_economy_tradeoff(reports, output_dir)  # 安全性-经济性权衡
                plot_key_risk_indicators(reports, output_dir)  # 关键风险指标雷达图
                plot_comprehensive_scorecard(reports, output_dir)  # 综合性能评分卡
                # --- 修改调用 ---
                # plot_detailed_confusion_matrix(y_true, y_pred_models, le.classes_, output_dir) # <--- 旧调用 ---
                # plot_key_class_performance_summary(reports, output_dir) # <--- 旧调用 ---
                # plot_model_comparison_charts(reports, output_dir) # <--- 旧调用 (错误的) ---
                # --- 新增调用 ---
                plot_detailed_confusion_matrix(y_true, y_pred_models, le.classes_, output_dir) # <--- 保持不变 ---
                # --- 修改：传递正确的参数 ---
                plot_key_class_performance_summary(y_pred_models, le.classes_, y_true, output_dir) # <--- 修改这里 ---
                plot_model_comparison_charts(y_pred_models, le.classes_, y_true, output_dir) # <--- 修改这里 ---
                # --- 修改结束 ---

            # 3. 类别分布饼图 (作为背景信息)
            plot_class_distribution(y_true, le.classes_, output_dir)

            # 4. 综合混淆矩阵热力图
            if y_pred_models: # <--- 修改：使用 y_pred_models ---
                plot_confusion_matrix_heatmap(y_true, y_pred_models, le.classes_, output_dir) # <--- 修改：使用 y_pred_models ---

        print(f"\n🎉 任务二额外可视化图表已全部生成并保存至: {os.path.abspath(output_dir)}")

    except FileNotFoundError as e:
        print(f"‼️ 错误：找不到所需的数据文件 {e.filename}。")
        print("请确保已完整运行前面的数据处理脚本")
    except Exception as e:
        print(f"‼️ 发生错误: {e}")
        import traceback

        traceback.print_exc()
