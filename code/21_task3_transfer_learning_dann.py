# 21_task3_transfer_learning_dann.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE

# --- 新增：导入概率校准相关库 ---
try:
    from netcal.scaling import TemperatureScaling

    HAS_NETCAL = True
except ImportError:
    HAS_NETCAL = False
    print("⚠️  未安装 netcal 库。请运行 'pip install netcal' 以启用 Temperature Scaling 概率校准。")
# --- 新增结束 ---
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import scipy.io
from scipy.signal import hilbert
import argparse
import datetime


# 设置 GPU 内存增长 (如果使用GPU)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# 设置中文字体
def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")


# 创建输出目录
def create_output_dir(base_name="task3_outputs_dann"):
    """创建输出目录"""
    output_dir = os.path.join('..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# --- 新增：定义梯度反转层 (Gradient Reversal Layer) ---
class GradientReversalLayer(Layer):
    """
    梯度反转层 (Gradient Reversal Layer, GRL)
    """

    def __init__(self, lambda_val=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lambda_val = lambda_val

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(GradientReversalLayer, self).get_config()
        config.update({'lambda_val': self.lambda_val})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# 自定义梯度函数
@tf.custom_gradient
def gradient_reversal(x, lambda_val):
    def grad(dy):
        return -lambda_val * dy, None

    return x, grad


class GradientReversalLayerCustom(Layer):
    """
    使用自定义梯度的梯度反转层
    """

    def __init__(self, lambda_val=1.0, **kwargs):
        super(GradientReversalLayerCustom, self).__init__(**kwargs)
        self.lambda_val = lambda_val

    def call(self, inputs):
        return gradient_reversal(inputs, self.lambda_val)

    def get_config(self):
        config = super(GradientReversalLayerCustom, self).get_config()
        config.update({'lambda_val': self.lambda_val})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# --- 新增结束 ---


# --- 新增：定义 DANN 模型 ---
def create_dann_model(input_dim, num_classes, lambda_grl=1.0, dropout_rate=0.5):
    """
    创建 DANN 模型。
    """
    # 1. 输入层
    inputs = Input(shape=(input_dim,), name='input')

    # 2. 特征提取器 (共享特征)
    shared = Dense(128, activation='relu', name='feature_extractor_1')(inputs)
    shared = Dropout(dropout_rate, name='feature_extractor_dropout_1')(shared)
    shared = Dense(64, activation='relu', name='feature_extractor_2')(shared)
    shared = Dropout(dropout_rate, name='feature_extractor_dropout_2')(shared)
    features_shared = Dense(32, activation='relu', name='feature_extractor_3')(shared)

    # --- 领域自适应分支 ---
    # 3a. 梯度反转层 (GRL)
    grl = GradientReversalLayerCustom(lambda_val=lambda_grl, name='grl')(features_shared)

    # 3b. 领域判别器 (Domain Discriminator)
    d_net = Dense(32, activation='relu', name='domain_discriminator_1')(grl)
    d_net = Dropout(dropout_rate, name='domain_discriminator_dropout_1')(d_net)
    domain_output = Dense(1, activation='sigmoid', name='domain_output')(d_net)
    # --- 领域自适应分支结束 ---

    # 4. 主任务分类头 (Task Classifier)
    c_net = Dense(64, activation='relu', name='classifier_1')(features_shared)
    c_net = Dropout(dropout_rate, name='classifier_dropout_1')(c_net)
    class_output = Dense(num_classes, activation='softmax', name='class_output')(c_net)

    # 5. 构建模型
    model = Model(inputs=inputs, outputs=[class_output, domain_output], name='DANN_Model')

    return model


# --- 新增结束 ---


# --- 新增：自定义训练循环 ---
def train_dann(model, X_s_train, y_s_train_cat, X_t_train, X_s_val, y_s_val_cat, epochs, batch_size, lambda_domain=1.0):
    """
    使用自定义训练循环训练 DANN 模型。
    """
    print("  - 开始自定义对抗训练 (领域自适应)...")

    # 准备 TensorFlow 数据集
    # 源域训练集 (用于分类和领域损失)
    ds_source_train = tf.data.Dataset.from_tensor_slices((X_s_train, y_s_train_cat))
    ds_source_train = ds_source_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 目标域训练集 (仅用于领域损失)
    domain_labels_target_train = tf.ones((X_t_train.shape[0], 1))
    ds_target_train = tf.data.Dataset.from_tensor_slices((X_t_train, domain_labels_target_train))
    ds_target_train = ds_target_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 源域验证集 (用于监控分类性能)
    ds_source_val = tf.data.Dataset.from_tensor_slices((X_s_val, y_s_val_cat))
    ds_source_val = ds_source_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # 损失函数
    classification_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    domain_loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Metrics
    train_class_acc = tf.keras.metrics.CategoricalAccuracy(name='train_class_accuracy')
    train_domain_acc = tf.keras.metrics.BinaryAccuracy(name='train_domain_accuracy')
    val_class_acc = tf.keras.metrics.CategoricalAccuracy(name='val_class_accuracy')

    @tf.function
    def train_step(x_s, y_s, x_t, lambda_d):
        # 源域数据域标签为 0，目标域数据域标签为 1
        domain_labels_s = tf.zeros((tf.shape(x_s)[0], 1))
        domain_labels_t = tf.ones((tf.shape(x_t)[0], 1))

        with tf.GradientTape() as tape:
            # 前向传播
            y_pred_s, d_pred_s = model(x_s, training=True)
            _, d_pred_t = model(x_t, training=True)

            # 计算损失
            class_loss_s = classification_loss_fn(y_s, y_pred_s)
            domain_loss_s = domain_loss_fn(domain_labels_s, d_pred_s)
            domain_loss_t = domain_loss_fn(domain_labels_t, d_pred_t)
            total_domain_loss = domain_loss_s + domain_loss_t
            total_loss = class_loss_s + lambda_d * total_domain_loss

        # 计算梯度并更新
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 更新指标
        train_class_acc.update_state(y_s, y_pred_s)
        combined_d_pred = tf.concat([d_pred_s, d_pred_t], axis=0)
        combined_d_labels = tf.concat([domain_labels_s, domain_labels_t], axis=0)
        train_domain_acc.update_state(combined_d_labels, combined_d_pred)

        return class_loss_s, total_domain_loss, total_loss

    @tf.function
    def val_step(x, y):
        y_pred, _ = model(x, training=False)
        val_class_acc.update_state(y, y_pred)

    # 训练循环
    history = {
        'epoch': [],
        'class_loss': [], 'domain_loss': [], 'total_loss': [],
        'train_acc': [], 'train_dom_acc': [], 'val_acc': []
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # 重置指标
        train_class_acc.reset_state()
        train_domain_acc.reset_state()

        # 训练
        total_cls_loss = 0
        total_dom_loss = 0
        num_batches = 0

        # 将目标域数据重复以匹配源域数据批次
        ds_target_train_cycled = ds_target_train.repeat()
        ds_train_combined = tf.data.Dataset.zip((ds_source_train, ds_target_train_cycled)).take(len(ds_source_train))

        for (x_s, y_s), (x_t, _) in ds_train_combined:
            cls_loss, dom_loss, total_loss = train_step(x_s, y_s, x_t, lambda_domain)
            total_cls_loss += cls_loss
            total_dom_loss += dom_loss
            num_batches += 1

        # 验证
        val_class_acc.reset_state()
        for x_val, y_val in ds_source_val:
            val_step(x_val, y_val)

        avg_cls_loss = total_cls_loss / num_batches
        avg_dom_loss = total_dom_loss / num_batches

        print(f" - Avg Class Loss: {avg_cls_loss:.4f}, Avg Domain Loss: {avg_dom_loss:.4f}, "
              f"Train Acc: {train_class_acc.result():.4f}, Train Dom Acc: {train_domain_acc.result():.4f}, "
              f"Val Acc: {val_class_acc.result():.4f}")

        # 记录历史
        history['epoch'].append(epoch + 1)
        history['class_loss'].append(avg_cls_loss)
        history['domain_loss'].append(avg_dom_loss)
        history['total_loss'].append(avg_cls_loss + lambda_domain * avg_dom_loss)
        history['train_acc'].append(train_class_acc.result())
        history['train_dom_acc'].append(train_domain_acc.result())
        history['val_acc'].append(val_class_acc.result())

    print("✅ 自定义对抗训练完成。")
    return history


# --- 新增结束 ---


# --- 新增：概率校准函数 ---
def calibrate_model_with_temperature_scaling(model, X_val, y_val_int):
    """
    使用 Temperature Scaling 校准模型输出的概率。
    """
    print("  - 正在校准模型概率 (Temperature Scaling)...")
    if not HAS_NETCAL:
        print("  ⚠️  未安装 netcal 库，跳过概率校准。")
        return None

    try:
        print("  ⚠️  警告：当前模型结构输出 Softmax 概率，无法直接应用 Temperature Scaling。")
        print("      建议修改模型，使其输出 logits，并重新训练。")
        print("      当前将跳过概率校准。")
        return None

    except Exception as e:
        print(f"  ⚠️  模型校准过程中出错: {e}")
        return None


def apply_calibration_if_available(calibrator, y_pred_proba):
    """
    如果校准器存在，则应用校准。
    """
    if calibrator is not None and HAS_NETCAL:
        print("  - 应用已训练的校准器...")
        try:
            y_pred_proba_calibrated = calibrator.transform(y_pred_proba)
            return y_pred_proba_calibrated
        except Exception as e:
            print(f"  ⚠️  应用校准器时出错: {e}。使用原始模型预测。")
            return y_pred_proba
    else:
        print("  - 未找到校准器或未安装 netcal，使用原始模型预测...")
        return y_pred_proba


# --- 新增结束 ---


# --- 新增：多次训练与投票预测函数 ---
def train_and_predict_with_voting(
        input_dim, num_classes, selected_feature_names,
        X_s_train, y_s_train_cat, X_t_train, X_s_val, y_s_val_cat,
        df_target_features, scaler, le,
        seeds=[42, 123, 456, 789, 999],  # 指定随机种子
        epochs=20, batch_size=64, lambda_domain=1.0, lambda_grl=1.0, dropout_rate=0.5
):
    """
    执行多次训练和预测，并收集结果用于投票。
    """
    print("  - 开始多次训练与预测以支持投票机制...")
    all_predictions_per_file_per_run = {}  # {run_id: {filename: [pred_label_int, ...]}}
    all_probabilities_per_file_per_run = {}  # {run_id: {filename: [pred_proba_array, ...]}}

    # 1. 准备目标域特征数据
    target_filenames = df_target_features['source_file'].values  # 假设列名是 source_file
    X_target_raw = df_target_features[selected_feature_names]
    X_target_scaled = scaler.transform(X_target_raw)

    for run_id, seed in enumerate(seeds):
        print(f"\n    --- 第 {run_id + 1}/{len(seeds)} 次训练 (种子: {seed}) ---")
        # 1. 设置随机种子
        tf.random.set_seed(seed)
        np.random.seed(seed)  # 影响 numpy 相关的随机操作

        # 2. 重新创建模型 (确保每次都是全新初始化)
        print(f"      - 正在构建第 {run_id + 1} 个 DANN 模型...")
        model = create_dann_model(input_dim, num_classes, lambda_grl, dropout_rate)

        # 3. 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                'class_output': 'categorical_crossentropy',
                'domain_output': 'binary_crossentropy'
            },
            loss_weights={
                'class_output': 1.0,
                'domain_output': lambda_domain
            },
            metrics={
                'class_output': 'accuracy',
                'domain_output': 'accuracy'
            }
        )

        # 4. 训练模型 (使用自定义循环)
        print(f"      - 正在训练第 {run_id + 1} 个 DANN 模型...")
        history = train_dann(model, X_s_train, y_s_train_cat, X_t_train, X_s_val, y_s_val_cat, epochs, batch_size,
                             lambda_domain)
        print(f"      ✅ 第 {run_id + 1} 个 DANN 模型训练完成。")

        # 5. 保存模型
        model_save_path = os.path.join('..', 'data', 'processed', f'dann_model_run_{run_id + 1}_seed_{seed}.h5')
        model.save(model_save_path)
        print(f"      ✅ 第 {run_id + 1} 个 DANN 模型已保存至: {model_save_path}")

        # 6. 构建用于预测的模型 (移除域输出)
        final_model = Model(inputs=model.input, outputs=model.get_layer('class_output').output)

        # 7. 预测目标域
        print(f"      - 正在对目标域数据进行第 {run_id + 1} 次预测...")
        y_target_pred_proba = final_model.predict(X_target_scaled)  # 形状: (N_samples, num_classes)
        y_target_pred_int = np.argmax(y_target_pred_proba, axis=1)  # 形状: (N_samples,)

        # 8. 存储预测结果 (按文件名分组)
        print(f"      - 正在存储第 {run_id + 1} 次预测结果...")
        for i, fname in enumerate(target_filenames):
            if run_id == 0:
                all_predictions_per_file_per_run[fname] = []
                all_probabilities_per_file_per_run[fname] = []
            all_predictions_per_file_per_run[fname].append(y_target_pred_int[i])
            all_probabilities_per_file_per_run[fname].append(y_target_pred_proba[i])  # 存储原始概率

        print(f"      ✅ 第 {run_id + 1} 次预测与存储完成。")

    # 9. 投票决策
    print("\n  - 正在根据多次预测结果进行文件级投票...")
    final_predictions = {}
    final_confidences = {}  # 存储投票后的置信度 (例如，获胜类别的平均概率)
    for fname in target_filenames:  # 遍历所有文件名
        predictions_list = all_predictions_per_file_per_run.get(fname, [])
        probabilities_list = all_probabilities_per_file_per_run.get(fname, [])
        if predictions_list and probabilities_list:
            # 简单多数投票
            values, counts = np.unique(predictions_list, return_counts=True)
            final_label_int = values[np.argmax(counts)]
            final_label_str = le.inverse_transform([final_label_int])[0]
            final_predictions[fname] = final_label_str

            # 计算置信度：获胜类别的平均概率
            winning_class_probs = [probs[final_label_int] for probs in probabilities_list]
            final_confidences[fname] = np.mean(winning_class_probs)

        else:
            final_predictions[fname] = 'Unknown'
            final_confidences[fname] = 0.0

    print("  ✅ 文件级投票完成。")

    return final_predictions, final_confidences


def load_and_predict_with_voting(
        input_dim, num_classes, selected_feature_names,
        df_target_features, scaler, le,
        seeds=[42, 123, 456, 789, 999],  # 指定随机种子
        lambda_grl=1.0, dropout_rate=0.5
):
    """
    加载已训练的模型并进行预测，然后投票。
    """
    print("  - 开始加载已训练模型并进行预测以支持投票机制...")
    all_predictions_per_file_per_run = {}  # {run_id: {filename: [pred_label_int, ...]}}
    all_probabilities_per_file_per_run = {}  # {run_id: {filename: [pred_proba_array, ...]}}

    # 1. 准备目标域特征数据
    target_filenames = df_target_features['source_file'].values  # 假设列名是 source_file
    X_target_raw = df_target_features[selected_feature_names]
    X_target_scaled = scaler.transform(X_target_raw)

    for run_id, seed in enumerate(seeds):
        print(f"\n    --- 加载第 {run_id + 1}/{len(seeds)} 个已训练模型 (种子: {seed}) ---")

        # 1. 加载模型
        model_load_path = os.path.join('..', 'data', 'processed', f'dann_model_run_{run_id + 1}_seed_{seed}.h5')
        try:
            model = tf.keras.models.load_model(model_load_path, custom_objects={
                'GradientReversalLayerCustom': GradientReversalLayerCustom})
            print(f"      ✅ 成功加载第 {run_id + 1} 个 DANN 模型: {model_load_path}")
        except Exception as e:
            print(f"      ❌ 加载模型失败: {e}")
            continue

        # 2. 构建用于预测的模型 (移除域输出)
        final_model = Model(inputs=model.input, outputs=model.get_layer('class_output').output)

        # 3. 预测目标域
        print(f"      - 正在对目标域数据进行第 {run_id + 1} 次预测...")
        y_target_pred_proba = final_model.predict(X_target_scaled)  # 形状: (N_samples, num_classes)
        y_target_pred_int = np.argmax(y_target_pred_proba, axis=1)  # 形状: (N_samples,)

        # 4. 存储预测结果 (按文件名分组)
        print(f"      - 正在存储第 {run_id + 1} 次预测结果...")
        for i, fname in enumerate(target_filenames):
            if run_id == 0:
                all_predictions_per_file_per_run[fname] = []
                all_probabilities_per_file_per_run[fname] = []
            all_predictions_per_file_per_run[fname].append(y_target_pred_int[i])
            all_probabilities_per_file_per_run[fname].append(y_target_pred_proba[i])  # 存储原始概率

        print(f"      ✅ 第 {run_id + 1} 次预测与存储完成。")

    # 5. 投票决策
    print("\n  - 正在根据多次预测结果进行文件级投票...")
    final_predictions = {}
    final_confidences = {}  # 存储投票后的置信度 (例如，获胜类别的平均概率)
    for fname in target_filenames:  # 遍历所有文件名
        predictions_list = all_predictions_per_file_per_run.get(fname, [])
        probabilities_list = all_probabilities_per_file_per_run.get(fname, [])
        if predictions_list and probabilities_list:
            # 简单多数投票
            values, counts = np.unique(predictions_list, return_counts=True)
            final_label_int = values[np.argmax(counts)]
            final_label_str = le.inverse_transform([final_label_int])[0]
            final_predictions[fname] = final_label_str

            # 计算置信度：获胜类别的平均概率
            winning_class_probs = [probs[final_label_int] for probs in probabilities_list]
            final_confidences[fname] = np.mean(winning_class_probs)

        else:
            final_predictions[fname] = 'Unknown'
            final_confidences[fname] = 0.0

    print("  ✅ 文件级投票完成。")

    return final_predictions, final_confidences


# --- 新增结束 ---


# --- 新增：t-SNE 可视化 (修正版) ---
def visualize_tsne(X_source, y_source, X_target, source_le, target_predictions_dict, output_dir, title_suffix=""):
    """
    使用 t-SNE 可视化源域和目标域数据 (修正版)。
    """
    print(f"  - 正在对特征进行 t-SNE 降维 ({title_suffix})...")
    try:
        # 合并特征
        all_features = np.vstack([X_source, X_target])
        # 创建域标签
        domain_labels = np.array(['Source'] * len(X_source) + ['Target'] * len(X_target))
        total_samples = len(all_features)

        # t-SNE 降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(all_features)

        plt.figure(figsize=(12, 10))

        # --- 修正：绘制源域数据（按类别着色） ---
        unique_source_classes = np.unique(y_source)
        # 使用 matplotlib.colormaps (推荐，避免弃用警告)
        try:
            colors_classes = plt.colormaps['tab10']
        except AttributeError:
            # 兼容旧版本 matplotlib
            colors_classes = plt.cm.get_cmap('tab10')

        colors_classes = colors_classes(np.linspace(0, 1, len(unique_source_classes)))

        for i, cls_int in enumerate(unique_source_classes):
            cls_name = source_le.inverse_transform([cls_int])[0]
            # 1. 找到所有源域样本在 tsne_results 中的索引
            source_indices = np.where(domain_labels == 'Source')[0]  # 形状 (len(X_source),)
            # 2. 在源域样本中，找到属于当前类别的样本索引 (相对于 y_source)
            within_source_class_indices_bool = (y_source == cls_int)  # 形状 (len(X_source),)
            # 3. 将 2 中的布尔索引应用到 1 中的索引上，得到最终在 tsne_results 中的索引
            final_indices = source_indices[within_source_class_indices_bool]  # 形状 (该类别的源域样本数,)

            if len(final_indices) > 0:
                plt.scatter(tsne_results[final_indices, 0], tsne_results[final_indices, 1],
                            c=[colors_classes[i]], label=f'Source-{cls_name}', alpha=0.6, s=20)
        # --- 修正结束 ---

        # --- 修正：绘制目标域数据（按预测类别着色）---
        target_filenames_in_order = list(target_predictions_dict.keys())
        y_target_pred_str = [target_predictions_dict[fname] for fname in target_filenames_in_order]

        # 处理目标域预测标签不在源域标签中的情况
        valid_target_mask = np.array([label in source_le.classes_ for label in y_target_pred_str])
        valid_target_filenames = np.array(target_filenames_in_order)[valid_target_mask]
        valid_y_target_pred_str = np.array(y_target_pred_str)[valid_target_mask]

        if len(valid_y_target_pred_str) > 0:
            try:
                y_target_pred_int_valid = source_le.transform(valid_y_target_pred_str)
                target_indices = np.where(domain_labels == 'Target')[0]  # 形状 (len(X_target),)
                valid_target_indices = target_indices[valid_target_mask]  # 形状 (len(valid_y_target_pred_str),)

                for i, cls_int in enumerate(unique_source_classes):
                    cls_name = source_le.inverse_transform([cls_int])[0]
                    # 找到目标域中预测为当前类别的样本索引
                    within_target_class_indices_bool = (
                            y_target_pred_int_valid == cls_int)  # 形状 (len(valid_y_target_pred_str),)
                    final_target_indices = valid_target_indices[within_target_class_indices_bool]  # 形状 (该类别预测的目标域样本数,)

                    if len(final_target_indices) > 0:
                        plt.scatter(tsne_results[final_target_indices, 0], tsne_results[final_target_indices, 1],
                                    c=[colors_classes[i]], label=f'Target-{cls_name}', alpha=0.5, s=20, marker='x')
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
        # --- 修正结束 ---

        plt.title(f't-SNE 可视化: 源域与目标域特征分布 {title_suffix}', fontsize=16)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        # 调整图例位置，避免超出边界
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'21_tsne_features_{title_suffix.replace(" ", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - ✅ t-SNE 图已保存至: {save_path}")
    except Exception as e:
        print(f"  - ⚠️ t-SNE 可视化失败: {e}")
        import traceback
        traceback.print_exc()


def visualize_tsne_before_after(X_source, y_source, X_target, source_le, target_predictions_dict,
                                trained_dann_model, output_dir):
    """
    可视化迁移前后的特征分布。
    """
    print("  - 开始迁移前后 t-SNE 可视化...")

    # 1. 迁移前：使用原始特征
    print("    - 生成迁移前 t-SNE 可视化...")
    visualize_tsne(X_source, y_source, X_target, source_le, target_predictions_dict, output_dir, "迁移前")

    # 2. 迁移后：使用 DANN 模型提取的特征
    print("    - 生成迁移后 t-SNE 可视化...")

    # 提取特征表示（移除分类头和域判别器）
    feature_extractor = Model(inputs=trained_dann_model.input,
                              outputs=trained_dann_model.get_layer('feature_extractor_3').output)

    X_source_features = feature_extractor.predict(X_source)
    X_target_features = feature_extractor.predict(X_target)

    # 重新进行 t-SNE 降维
    all_features_extracted = np.vstack([X_source_features, X_target_features])
    domain_labels = np.array(['Source'] * len(X_source_features) + ['Target'] * len(X_target_features))

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(all_features_extracted)

    plt.figure(figsize=(12, 10))

    # 绘制源域数据（按类别着色）
    unique_source_classes = np.unique(y_source)
    try:
        colors_classes = plt.colormaps['tab10']
    except AttributeError:
        colors_classes = plt.cm.get_cmap('tab10')

    colors_classes = colors_classes(np.linspace(0, 1, len(unique_source_classes)))

    for i, cls_int in enumerate(unique_source_classes):
        cls_name = source_le.inverse_transform([cls_int])[0]
        source_indices = np.where(domain_labels == 'Source')[0]
        within_source_class_indices_bool = (y_source == cls_int)
        final_indices = source_indices[within_source_class_indices_bool]

        if len(final_indices) > 0:
            plt.scatter(tsne_results[final_indices, 0], tsne_results[final_indices, 1],
                        c=[colors_classes[i]], label=f'Source-{cls_name}', alpha=0.6, s=20)

    # 绘制目标域数据（按预测类别着色）
    target_filenames_in_order = list(target_predictions_dict.keys())
    y_target_pred_str = [target_predictions_dict[fname] for fname in target_filenames_in_order]
    valid_target_mask = np.array([label in source_le.classes_ for label in y_target_pred_str])
    valid_y_target_pred_str = np.array(y_target_pred_str)[valid_target_mask]

    if len(valid_y_target_pred_str) > 0:
        try:
            y_target_pred_int_valid = source_le.transform(valid_y_target_pred_str)
            target_indices = np.where(domain_labels == 'Target')[0]
            valid_target_indices = target_indices[valid_target_mask]

            for i, cls_int in enumerate(unique_source_classes):
                cls_name = source_le.inverse_transform([cls_int])[0]
                within_target_class_indices_bool = (y_target_pred_int_valid == cls_int)
                final_target_indices = valid_target_indices[within_target_class_indices_bool]

                if len(final_target_indices) > 0:
                    plt.scatter(tsne_results[final_target_indices, 0], tsne_results[final_target_indices, 1],
                                c=[colors_classes[i]], label=f'Target-{cls_name}', alpha=0.5, s=20, marker='x')
        except ValueError as e:
            print(f"  ⚠️  标签转换错误: {e}")
            target_indices = np.where(domain_labels == 'Target')[0]
            plt.scatter(tsne_results[target_indices, 0], tsne_results[target_indices, 1],
                        c='red', label='Target (Uncertain Labels)', alpha=0.5, s=20, marker='x')
    else:
        print("  ⚠️  没有有效的目标域预测标签用于可视化。")
        target_indices = np.where(domain_labels == 'Target')[0]
        plt.scatter(tsne_results[target_indices, 0], tsne_results[target_indices, 1],
                    c='red', label='Target (No Valid Labels)', alpha=0.5, s=20, marker='x')

    plt.title('t-SNE 可视化: 源域与目标域特征分布 (迁移后 - DANN提取特征)', fontsize=16)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, '21_tsne_features_迁移后.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - ✅ 迁移后 t-SNE 图已保存至: {save_path}")


# --- 新增结束 ---


# 主程序
if __name__ == "__main__":
    set_chinese_font()
    output_dir = create_output_dir()

    print("🚀 任务三：迁移诊断 (基于 DANN 和领域自适应) 开始执行...")

    # --- 阶段一：加载源域模型与预处理器 ---
    print("\n--- 阶段一：加载源域模型与预处理器 ---")

    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    TASK2_OUTPUTS_DIR = os.path.join(PROCESSED_DIR, 'task2_outputs_final')  # 假设模型保存在此目录

    SCALER_PATH = os.path.join(TASK2_OUTPUTS_DIR, 'final_scaler.joblib')
    LE_PATH = os.path.join(TASK2_OUTPUTS_DIR, 'final_label_encoder.joblib')
    # 注意：DANN 模型权重将在此脚本中训练并保存

    try:
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(LE_PATH)
        print("✅ 成功加载 StandardScaler 和 LabelEncoder。")
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到文件 {e.filename}。请确保任务二已成功运行并生成了输出文件。")
        exit(1)
    except Exception as e:
        print(f"❌ 加载预处理器时发生错误: {e}")
        import traceback

        traceback.print_exc()  # 打印完整的错误堆栈
        exit(1)

    # --- 阶段二：加载并预处理源域和目标域数据 ---
    print("\n--- 阶段二：加载并预处理源域和目标域数据 ---")

    # 1. 加载源域数据 (用于训练迁移模型)
    SOURCE_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features_selected.csv')
    TARGET_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'target_features.csv')  # 假设这是06脚本处理后的特征

    try:
        df_source_features = pd.read_csv(SOURCE_FEATURES_PATH)
        df_target_features = pd.read_csv(TARGET_FEATURES_PATH)
        print(f"✅ 成功加载源域特征数据: {df_source_features.shape}")
        print(f"✅ 成功加载目标域特征数据: {df_target_features.shape}")

        # 2. 分离特征和标签 (源域)
        selected_feature_names = df_source_features.drop(columns=['label', 'rpm', 'filename']).columns.tolist()
        X_source_raw = df_source_features[selected_feature_names]
        y_source_str = df_source_features['label']
        y_source = le.transform(y_source_str)  # 转换为整数标签
        num_classes = len(le.classes_)
        y_source_cat = to_categorical(y_source, num_classes=num_classes)  # 转换为 one-hot

        # 3. 分离文件名和特征 (目标域)
        # 确保 df_target_features 包含 'source_file' 列，标识每个样本来自哪个 .mat 文件
        # 这需要在 06 脚本中实现滑动窗口并记录来源
        assert 'source_file' in df_target_features.columns, "目标域特征数据缺少 'source_file' 列!"
        target_filenames = df_target_features['source_file']
        X_target_raw = df_target_features[selected_feature_names]  # 确保使用相同的特征

        # 4. 标准化特征
        X_source_scaled = scaler.transform(X_source_raw)
        X_target_scaled = scaler.transform(X_target_raw)

        input_dim = X_source_scaled.shape[1]
        print(f"✅ 数据预处理完成。输入维度: {input_dim}")

    except FileNotFoundError as e:
        print(f"❌ 错误：找不到数据文件 {e.filename}。")
        exit(1)
    except Exception as e:
        print(f"❌ 处理数据时发生错误: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # --- 阶段三：深入分析源域与目标域的共性与差异 (可视化) ---
    print("\n--- 阶段三：深入分析源域与目标域的共性与差异 ---")
    # (此部分可以保持不变或根据新特征调整，暂时省略以聚焦核心训练)

    # --- 阶段四：设计并训练迁移模型 (领域自适应) ---
    print("\n--- 阶段四：设计并训练迁移模型 (领域自适应) ---")
    try:
        print("  - 构建包含领域自适应的 DANN 模型...")
        lambda_grl = 1.0  # GRL 的 lambda 参数
        dropout_rate = 0.5
        # dann_model = create_dann_model(input_dim, num_classes, lambda_grl=lambda_grl, dropout_rate=dropout_rate)
        # dann_model.summary()
        # print("✅ DANN 模型构建完成。")

        # --- 关键修改：划分源域训练/验证集 ---
        print("  - 划分源域训练集和验证集...")
        from sklearn.model_selection import train_test_split

        X_s_train, X_s_val, y_s_train, y_s_val, y_s_train_cat, y_s_val_cat = train_test_split(
            X_source_scaled, y_source, y_source_cat, test_size=0.2, random_state=42, stratify=y_source
        )
        X_t_train = X_target_scaled  # 目标域数据全部用于训练
        print(f"    - 源域训练集: {X_s_train.shape}, 验证集: {X_s_val.shape}")
        print(f"    - 目标域训练集: {X_t_train.shape}")

        # --- 关键修改：使用多次训练和投票 ---
        print("\n--- 启动多次训练与投票预测流程 ---")
        final_predictions, final_confidences = train_and_predict_with_voting(
            input_dim=input_dim,
            num_classes=num_classes,
            selected_feature_names=selected_feature_names,
            X_s_train=X_s_train, y_s_train_cat=y_s_train_cat,
            X_t_train=X_t_train,
            X_s_val=X_s_val, y_s_val_cat=y_s_val_cat,
            df_target_features=df_target_features,
            scaler=scaler, le=le,
            seeds=[42, 123, 456, 789, 999],  # 使用指定种子
            epochs=20,  # 可根据需要调整
            batch_size=64,
            lambda_domain=1.0,
            lambda_grl=lambda_grl,
            dropout_rate=dropout_rate
        )
        print("✅ 多次训练与投票预测流程完成。")

        # --- 阶段五：目标域预测与标定 ---
        print("\n--- 阶段五：目标域预测与标定 ---")
        # final_model = Model(inputs=dann_model.input, outputs=dann_model.get_layer('class_output').output)
        # y_target_pred_proba = final_model.predict(X_target_scaled)
        # y_target_pred_int = np.argmax(y_target_pred_proba, axis=1)
        # y_target_pred_labels = le.inverse_transform(y_target_pred_int)
        # print("✅ 目标域数据预测完成。")

        # --- 使用投票结果 ---
        # final_predictions 和 final_confidences 已经是文件级的最终预测结果和置信度
        print("✅ 目标域数据预测与标定完成 (基于投票)。")

        # --- 阶段六：迁移结果可视化展示与分析 ---
        print("\n--- 阶段六：迁移结果可视化展示与分析 ---")
        # 1. 预测标签分布
        # unique_labels, counts = np.unique(y_target_pred_labels, return_counts=True)
        unique_labels, counts = np.unique(list(final_predictions.values()), return_counts=True)
        plt.figure(figsize=(8, 6))
        colors = sns.color_palette("husl", len(unique_labels))
        plt.pie(counts, labels=unique_labels, autopct='%1.1f%%', startangle=140, colors=colors)
        # plt.title('目标域预测结果类别分布 (DANN)', fontsize=14, weight='bold')
        plt.title('目标域预测结果类别分布 (DANN + 投票)', fontsize=14, weight='bold')  # 更新标题
        plt.axis('equal')
        # save_path_pie = os.path.join(output_dir, '21_target_prediction_distribution_dann.png')
        save_path_pie = os.path.join(output_dir, '21_target_prediction_distribution_dann_voted.png')  # 更新文件名
        plt.savefig(save_path_pie, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 预测标签分布饼图已保存至: {save_path_pie}")

        # 2. 预测置信度分析
        # max_probs = np.max(y_target_pred_proba, axis=1)
        max_probs = list(final_confidences.values())  # 使用投票后的置信度
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('预测置信度 (投票后获胜类平均概率)', fontsize=12)  # 更新xlabel
        plt.ylabel('文件数量', fontsize=12)  # 更新ylabel
        # plt.title('目标域预测置信度分布 (DANN)', fontsize=14, weight='bold')
        plt.title('目标域预测置信度分布 (DANN + 投票)', fontsize=14, weight='bold')  # 更新标题
        plt.grid(True, alpha=0.3)
        mean_conf = np.mean(max_probs)
        median_conf = np.median(max_probs)
        stats_text = f'均值: {mean_conf:.3f}\n中位数: {median_conf:.3f}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        # save_path_hist = os.path.join(output_dir, '21_prediction_confidence_histogram_dann.png')
        save_path_hist = os.path.join(output_dir, '21_prediction_confidence_histogram_dann_voted.png')  # 更新文件名
        plt.savefig(save_path_hist, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 预测置信度直方图已保存至: {save_path_hist}")
        print(f"📊 预测置信度统计 - 均值: {mean_conf:.4f}, 中位数: {median_conf:.4f}")

        # 3. 保存预测结果
        # results_df = pd.DataFrame({
        #     'filename': target_filenames,
        #     'predicted_label': y_target_pred_labels,
        #     'confidence': max_probs
        # })
        # results_df = results_df.sort_values(by='filename').reset_index(drop=True)
        # RESULTS_CSV_PATH = os.path.join(output_dir, '21_target_domain_predictions_dann.csv')
        results_df = pd.DataFrame({
            'filename': list(final_predictions.keys()),
            'predicted_label': list(final_predictions.values()),
            'confidence': [final_confidences[fname] for fname in final_predictions.keys()]  # 添加置信度
        })
        results_df = results_df.sort_values(by='filename').reset_index(drop=True)
        RESULTS_CSV_PATH = os.path.join(output_dir, '21_target_domain_predictions_dann_voted.csv')  # 更新文件名
        results_df.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"✅ 目标域预测结果已保存至: {RESULTS_CSV_PATH}")

        # 4. (可选) t-SNE 可视化 (迁移前后对比)
        # 加载最后一个训练的模型用于可视化
        last_model_path = os.path.join('..', 'data', 'processed', f'dann_model_run_5_seed_999.h5')
        try:
            trained_dann_model = tf.keras.models.load_model(last_model_path, custom_objects={
                'GradientReversalLayerCustom': GradientReversalLayerCustom})
            visualize_tsne_before_after(X_source_scaled, y_source, X_target_scaled, le,
                                        final_predictions, trained_dann_model, output_dir)
        except Exception as e:
            print(f"⚠️  加载模型进行t-SNE可视化失败: {e}")
            # 退回到原始可视化
            visualize_tsne(X_source_scaled, y_source, X_target_scaled, le,
                           final_predictions,  # 使用投票后的预测结果
                           output_dir, title_suffix="迁移后 (投票)")

        print(f"\n🏆 任务三迁移诊断完成 (DANN + 领域自适应 + 投票)!")  # 更新最终打印信息
        print(f"   - 使用的基础模型: DANN (自定义训练循环)")
        print(f"   - 迁移策略: 领域自适应 (Domain Adaptation with GRL)")
        print(f"   - 预测策略: 多次训练 (5 seeds) + 投票决策")  # 更新信息
        print(f"   - 预测结果已保存在: {RESULTS_CSV_PATH}")
        print(f"   - 可视化图表已保存在: {os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"❌ 在模型构建、训练或预测阶段发生错误: {e}")
        import traceback

        traceback.print_exc()
        exit(1)