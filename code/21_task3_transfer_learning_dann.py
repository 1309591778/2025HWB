# 21_task3_transfer_learning_dann.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
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


# --- 新增：定义包含领域自适应的 DANN 模型架构 ---
def create_improved_dann_model(input_dim, num_classes, lambda_grl=0.01, dropout_rate=0.3):
    """
    改进的DANN模型，使用更小的lambda值和更深的特征提取器
    """
    inputs = Input(shape=(input_dim,), name='input')

    # 更深的特征提取器
    shared = Dense(256, activation='relu', name='feature_extractor_1')(inputs)
    shared = Dropout(dropout_rate, name='feature_extractor_dropout_1')(shared)
    shared = Dense(128, activation='relu', name='feature_extractor_2')(shared)
    shared = Dropout(dropout_rate, name='feature_extractor_dropout_2')(shared)
    shared = Dense(64, activation='relu', name='feature_extractor_3')(shared)
    shared = Dropout(dropout_rate, name='feature_extractor_dropout_3')(shared)
    features_before_grl = Dense(32, activation='relu', name='feature_extractor_4')(shared)

    # 领域分支
    grl = GradReverse(lambda_val=lambda_grl)(features_before_grl)
    d_net = Dense(32, activation='relu')(grl)
    d_net = Dropout(dropout_rate)(d_net)
    domain_output = Dense(1, activation='sigmoid', name='domain_output')(d_net)

    # 分类分支
    c_net = Dense(64, activation='relu')(features_before_grl)
    c_net = Dropout(dropout_rate)(c_net)
    class_output = Dense(num_classes, activation='softmax', name='class_output')(c_net)

    model = Model(inputs=inputs, outputs=[class_output, domain_output], name='Improved_DANN_Model')
    return model


# --- 新增结束 ---


# --- 新增：自定义训练循环 (修正版) ---
def train_improved_dann(model, X_s_train, y_s_train_cat, X_t_train, X_s_val, y_s_val_cat,
                        epochs, batch_size, lambda_domain=0.001):
    """
    改进的DANN训练函数，使用更稳定的训练策略
    """
    print("  - 开始改进的对抗训练...")

    # 准备数据集，确保长度匹配
    ds_source_train = tf.data.Dataset.from_tensor_slices((X_s_train, y_s_train_cat))
    ds_source_train = ds_source_train.batch(batch_size).repeat().shuffle(1000).prefetch(tf.data.AUTOTUNE)

    ds_target_train = tf.data.Dataset.from_tensor_slices(X_t_train)
    ds_target_train = ds_target_train.batch(batch_size).repeat().shuffle(1000).prefetch(tf.data.AUTOTUNE)

    # 使用zip并限制步数
    steps_per_epoch = min(len(X_s_train) // batch_size, len(X_t_train) // batch_size)
    ds_combined = tf.data.Dataset.zip((ds_source_train, ds_target_train))

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # 降低学习率
    classification_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    domain_loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Metrics
    train_class_acc = tf.keras.metrics.CategoricalAccuracy(name='train_class_accuracy')
    train_domain_acc = tf.keras.metrics.BinaryAccuracy(name='train_domain_accuracy')
    val_class_acc = tf.keras.metrics.CategoricalAccuracy(name='val_class_accuracy')

    # 添加早停机制
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # 重置metrics
        train_class_acc.reset_state()
        train_domain_acc.reset_state()

        total_cls_loss = 0
        total_dom_loss = 0
        num_batches = 0

        for step, ((x_s, y_s), x_t) in enumerate(ds_combined.take(steps_per_epoch)):
            # 为目标域创建标签
            domain_labels_s = tf.zeros((tf.shape(x_s)[0], 1))  # 源域为0
            domain_labels_t = tf.ones((tf.shape(x_t)[0], 1))  # 目标域为1

            with tf.GradientTape() as tape:
                # 源域前向传播
                y_pred_s, d_pred_s = model(x_s, training=True)
                class_loss_s = classification_loss_fn(y_s, y_pred_s)
                domain_loss_s = domain_loss_fn(domain_labels_s, d_pred_s)

                # 目标域前向传播
                _, d_pred_t = model(x_t, training=True)
                domain_loss_t = domain_loss_fn(domain_labels_t, d_pred_t)

                # 总损失 - 关键修改：大幅增加分类损失权重，降低域对抗权重
                total_class_loss = class_loss_s
                total_domain_loss = domain_loss_s + domain_loss_t
                # 重要：确保分类任务不被域对抗任务压制
                total_loss = 5.0 * total_class_loss + lambda_domain * total_domain_loss

            # 更新参数
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 更新metrics
            train_class_acc.update_state(y_s, y_pred_s)
            combined_d_pred = tf.concat([d_pred_s, d_pred_t], axis=0)
            combined_d_labels = tf.concat([domain_labels_s, domain_labels_t], axis=0)
            train_domain_acc.update_state(combined_d_labels, combined_d_pred)

            total_cls_loss += total_class_loss
            total_dom_loss += total_domain_loss
            num_batches += 1

        # 验证
        val_class_acc.reset_state()
        val_pred = model(X_s_val, training=False)[0]  # 只取分类输出
        val_class_acc.update_state(y_s_val_cat, val_pred)

        avg_cls_loss = total_cls_loss / num_batches if num_batches > 0 else 0
        avg_dom_loss = total_dom_loss / num_batches if num_batches > 0 else 0

        current_val_acc = val_class_acc.result()
        print(f" - Cls Loss: {avg_cls_loss:.4f}, Dom Loss: {avg_dom_loss:.4f}, "
              f"Train Cls Acc: {train_class_acc.result():.4f}, Train Dom Acc: {train_domain_acc.result():.4f}, "
              f"Val Acc: {current_val_acc:.4f}")

        # 早停机制
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  - 早停触发：验证准确率连续{patience}轮未提升，停止训练")
                break

    print("✅ 改进的对抗训练完成。")


# --- 新增结束 ---


# 主程序
if __name__ == "__main__":
    set_chinese_font()
    output_dir = create_output_dir()

    print("🚀 任务三：迁移诊断 (基于 DANN 和领域自适应) 开始执行...")

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

    # --- 阶段二：加载并预处理源域和目标域数据 ---
    print("\n--- 阶段二：加载并预处理源域和目标域数据 ---")

    SOURCE_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features_selected.csv')
    TARGET_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'target_features.csv')

    try:
        df_source_features = pd.read_csv(SOURCE_FEATURES_PATH)
        df_target_features = pd.read_csv(TARGET_FEATURES_PATH)
        print(f"✅ 成功加载源域特征数据: {df_source_features.shape}")
        print(f"✅ 成功加载目标域特征数据: {df_target_features.shape}")

        selected_feature_names = df_source_features.drop(columns=['label', 'rpm', 'filename']).columns.tolist()
        X_source_raw = df_source_features[selected_feature_names]
        y_source_str = df_source_features['label']
        y_source = le.transform(y_source_str)
        num_classes = len(le.classes_)
        y_source_cat = to_categorical(y_source, num_classes=num_classes)

        X_target_raw = df_target_features[selected_feature_names]

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

    # --- 阶段四：设计并训练迁移模型 (领域自适应) ---
    print("\n--- 阶段四：设计并训练迁移模型 (领域自适应) ---")

    # 使用5个随机种子训练5个模型
    random_seeds = [42, 123, 456, 789, 999]

    for seed_idx, seed in enumerate(random_seeds):
        print(f"\n--- 训练第 {seed_idx + 1} 个模型 (随机种子: {seed}) ---")

        tf.random.set_seed(seed)
        np.random.seed(seed)

        try:
            print("  - 构建包含领域自适应的 DANN 模型...")
            lambda_grl = 0.01  # 关键：更小的 GRL lambda
            dropout_rate = 0.3
            dann_model = create_improved_dann_model(input_dim, num_classes, lambda_grl=lambda_grl,
                                                    dropout_rate=dropout_rate)
            print("✅ DANN 模型构建完成。")

            # 划分源域训练/验证集
            print("  - 划分源域训练集和验证集...")
            X_s_train, X_s_val, y_s_train, y_s_val, y_s_train_cat, y_s_val_cat = train_test_split(
                X_source_scaled, y_source, y_source_cat, test_size=0.2, random_state=42, stratify=y_source
            )
            X_t_train = X_target_scaled
            print(f"    - 源域训练集: {X_s_train.shape}, 验证集: {X_s_val.shape}")
            print(f"    - 目标域训练集: {X_t_train.shape}")

            # 训练模型
            print("  - 开始对抗训练 (领域自适应)...")
            epochs = 20
            batch_size = 64
            lambda_domain = 0.001  # 关键：极低的 domain loss 权重
            train_improved_dann(dann_model, X_s_train, y_s_train_cat, X_t_train, X_s_val, y_s_val_cat, epochs,
                                batch_size, lambda_domain)
            print("✅ 对抗训练完成。")

            # 保存训练好的模型
            model_save_path = os.path.join(output_dir, f'dann_model_seed_{seed}.h5')
            dann_model.save(model_save_path)
            print(f"✅ 模型已保存至: {model_save_path}")

        except Exception as e:
            print(f"❌ 在第 {seed_idx + 1} 个模型训练阶段发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n🏆 任务三迁移诊断完成 (DANN + 领域自适应)！")
    print(f"   - 已训练并保存 5 个 DANN 模型（不同随机种子）")
    print(f"   - 模型保存路径: {os.path.abspath(output_dir)}")