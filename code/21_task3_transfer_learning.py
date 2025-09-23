# 21_task3_transfer_learning.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import font_manager
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda  # 导入 MLP-DA 需要的层
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


# --- 新增：导入 GradReverse 层定义 ---
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
        self.lambda_val = lambda_val

    def call(self, x):
        return grad_reverse(x, self.lambda_val)

    def get_config(self):
        config = super(GradReverse, self).get_config()
        config.update({'lambda_val': self.lambda_val})
        return config


# --- 新增结束 ---


# 设置中文字体
def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")


# 创建输出目录
def create_output_dir():
    """创建输出目录"""
    # --- 修改：更新输出目录名称以反映模型变化 ---
    output_dir = os.path.join('..', 'data', 'processed', 'task3_outputs_mlp_da')  # <--- 修改这里
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# --- 修改：定义与任务二完全一致的 MLP-DA 模型架构 ---
def create_transfer_model(input_dim, num_classes, lambda_grl=1.0):  # <--- 修改函数签名
    """
    创建用于迁移学习的 MLP 模型，包含领域自适应组件。
    """
    # 1. 输入层
    inputs = Input(shape=(input_dim,))  # <--- 修改输入形状

    # 2. 特征提取器 (MLP)
    shared = Dense(128, activation='relu', name='feature_extractor_1')(inputs)
    shared = Dropout(0.5)(shared)
    shared = Dense(64, activation='relu', name='feature_extractor_2')(shared)
    shared = Dropout(0.5)(shared)
    features_before_grl = Dense(32, activation='relu', name='feature_extractor_3')(shared)  # <--- 修改特征层

    # --- 领域自适应分支 ---
    grl = GradReverse(lambda_val=lambda_grl)(features_before_grl)  # <--- 使用特征层
    d_net = Dense(32, activation='relu')(grl)
    d_net = Dropout(0.5)(d_net)
    domain_output = Dense(1, activation='sigmoid', name='domain_output')(d_net)
    # --- 领域自适应分支结束 ---

    # 4. 主任务分类头
    c_net = Dense(64, activation='relu')(features_before_grl)  # <--- 使用特征层
    c_net = Dropout(0.5)(c_net)
    class_output = Dense(num_classes, activation='softmax', name='class_output')(c_net)

    model = Model(inputs=inputs, outputs=[class_output, domain_output])

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # <--- 可能调整学习率
        loss={
            'class_output': 'categorical_crossentropy',
            'domain_output': 'binary_crossentropy'
        },
        loss_weights={
            'class_output': 1.0,
            'domain_output': 1.0  # lambda_domain
        },
        metrics={
            'class_output': 'accuracy',
            'domain_output': 'accuracy'
        }
    )
    return model


# --- 修改结束 ---


## --- 新增：自定义训练循环 (为 MLP-DA 调整) ---
def train_da_model(model, X_s_train, y_s_train_cat, X_t_train, X_s_val, y_s_val_cat, epochs, batch_size):
    """
    使用自定义训练循环训练领域自适应的 MLP 模型。
    """
    print("  - 开始自定义对抗训练 (领域自适应)...")

    # 准备 TensorFlow 数据集
    ds_source_train = tf.data.Dataset.from_tensor_slices((X_s_train, y_s_train_cat))
    ds_source_train = ds_source_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_target_train = tf.data.Dataset.from_tensor_slices((X_t_train, tf.ones((X_t_train.shape[0], 1))))
    ds_target_train = ds_target_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_source_val = tf.data.Dataset.from_tensor_slices((X_s_val, y_s_val_cat))
    ds_source_val = ds_source_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)  # <--- 与模型编译保持一致

    # 损失函数
    classification_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    domain_loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Metrics
    train_class_acc = tf.keras.metrics.CategoricalAccuracy(name='train_class_accuracy')
    train_domain_acc = tf.keras.metrics.BinaryAccuracy(name='train_domain_accuracy')
    val_class_acc = tf.keras.metrics.CategoricalAccuracy(name='val_class_accuracy')

    @tf.function
    def train_step(x_s, y_s, x_t):
        domain_labels_s = tf.zeros((tf.shape(x_s)[0], 1))
        domain_labels_t = tf.ones((tf.shape(x_t)[0], 1))

        with tf.GradientTape(persistent=True) as tape:
            y_pred_s, d_pred_s = model(x_s, training=True)
            class_loss_s = classification_loss_fn(y_s, y_pred_s)
            domain_loss_s = domain_loss_fn(domain_labels_s, d_pred_s)

            _, d_pred_t = model(x_t, training=True)
            domain_loss_t = domain_loss_fn(domain_labels_t, d_pred_t)

            total_class_loss = class_loss_s
            total_domain_loss = domain_loss_s + domain_loss_t
            total_loss = total_class_loss + total_domain_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_class_acc.update_state(y_s, y_pred_s)
        train_domain_acc.update_state(tf.concat([domain_labels_s, domain_labels_t], axis=0),
                                      tf.concat([d_pred_s, d_pred_t], axis=0))

        return total_class_loss, total_domain_loss

    @tf.function
    def val_step(x, y):
        y_pred, _ = model(x, training=False)
        val_class_acc.update_state(y, y_pred)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        # --- 关键修改：使用 reset_state() ---
        train_class_acc.reset_state()  # <--- 修改这里
        train_domain_acc.reset_state()  # <--- 修改这里
        # --- 关键修改结束 ---

        # 注意：直接 zip 不同长度的 dataset 可能导致较短的 dataset 用完后停止。
        # 为确保源域数据被充分利用，可以使用 cycle 或者确保 ds_source_train 是较长的那个。
        # 这里我们简单处理，假设源域数据足够多或与目标域批次对齐。
        # 如果需要更精确的控制，可以使用 tf.data.experimental.sample_from_datasets 或其他策略。
        num_batches_source = len(ds_source_train)
        num_batches_target = len(ds_target_train)
        num_batches = max(num_batches_source, num_batches_target)

        # 创建 zip dataset
        # repeat() 确保较短的数据集在需要时重复
        ds_source_train_repeat = ds_source_train.repeat()
        ds_target_train_repeat = ds_target_train.repeat()
        ds_train_combined = tf.data.Dataset.zip((ds_source_train_repeat, ds_target_train_repeat)).take(num_batches)

        total_cls_loss = 0
        total_dom_loss = 0
        num_steps = 0
        for (x_s, y_s), (x_t, _) in ds_train_combined:
            cls_loss, dom_loss = train_step(x_s, y_s, x_t)
            total_cls_loss += cls_loss
            total_dom_loss += dom_loss
            num_steps += 1

        # --- 关键修改：使用 reset_state() ---
        val_class_acc.reset_state()  # <--- 修改这里
        # --- 关键修改结束 ---
        for x_val, y_val in ds_source_val:
            val_step(x_val, y_val)

        # 计算平均损失进行打印
        avg_cls_loss = total_cls_loss / num_steps if num_steps > 0 else 0
        avg_dom_loss = total_dom_loss / num_steps if num_steps > 0 else 0

        print(f" - Avg Class Loss: {avg_cls_loss:.4f}, Avg Domain Loss: {avg_dom_loss:.4f}, "
              f"Train Acc: {train_class_acc.result():.4f}, Train Dom Acc: {train_domain_acc.result():.4f}, "
              f"Val Acc: {val_class_acc.result():.4f}")

    print("✅ 自定义对抗训练完成。")
# --- 新增结束 ---


# 主程序
if __name__ == "__main__":
    set_chinese_font()
    output_dir = create_output_dir()

    # --- 修改：更新打印信息 ---
    print("🚀 任务三：迁移诊断 (基于 MLP-DA 和领域自适应) 开始执行...")  # <--- 修改这里

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
    # --- 修改：指向聚合特征文件 ---
    SOURCE_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features_selected.csv')  # <--- 保持不变，因为是聚合特征
    TARGET_FEATURES_PATH = os.path.join(PROCESSED_DIR, 'target_features.csv')  # <--- 保持不变，因为是聚合特征

    try:
        # --- 修改：加载源域和目标域数据 (聚合特征) ---
        print("  - 正在加载源域聚合特征数据...")
        df_source_features = pd.read_csv(SOURCE_FEATURES_PATH)
        print("  - 正在加载目标域聚合特征数据...")
        df_target_features = pd.read_csv(TARGET_FEATURES_PATH)
        print(f"✅ 成功加载源域特征数据 (聚合): {df_source_features.shape}")
        print(f"✅ 成功加载目标域特征数据 (聚合): {df_target_features.shape}")

        # --- 修改：处理源域数据 (聚合特征) ---
        X_source_raw = df_source_features.drop(columns=['label', 'rpm', 'filename'])  # <--- 获取特征列
        y_source_str = df_source_features['label']
        y_source = le.transform(y_source_str)
        num_classes = len(le.classes_)
        y_source_cat = to_categorical(y_source, num_classes=num_classes)
        print(f"✅ 源域聚合特征数据形状: {X_source_raw.shape}")  # <--- 更新打印信息

        # --- 修改：处理目标域数据 (聚合特征) ---
        target_filenames = df_target_features['source_file']
        X_target_raw = df_target_features.drop(columns=['source_file', 'rpm'])  # <--- 获取特征列
        print(f"✅ 目标域聚合特征数据形状: {X_target_raw.shape}")  # <--- 更新打印信息

        # --- 修改：标准化 (直接对2D特征进行) ---
        X_source_scaled = scaler.transform(X_source_raw)  # Shape: (N_source, F)
        X_target_scaled = scaler.transform(X_target_raw)  # Shape: (N_target, F)

        input_dim = X_source_scaled.shape[1]  # <--- 获取输入维度 (特征数)
        print(f"✅ 数据标准化完成。输入维度: {input_dim}")  # <--- 更新打印信息

    except FileNotFoundError as e:
        print(f"❌ 错误：找不到数据文件 {e.filename}。")
        exit(1)
    except Exception as e:
        print(f"❌ 处理数据时发生错误: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # --- 阶段三：深入分析源域与目标域的共性与差异 (可视化) ---
    # (此部分可以保持不变或根据新特征调整，暂时省略以聚焦核心训练)

    # --- 阶段四：设计并训练迁移模型 (领域自适应) ---
    print("\n--- 阶段四：设计并训练迁移模型 (领域自适应) ---")
    try:
        # --- 修改：构建 MLP-DA 模型 ---
        print("  - 构建包含领域自适应的迁移模型 (MLP-DA)...")
        lambda_grl = 1.0
        # transfer_model = create_transfer_model(input_shape, num_classes, lambda_grl=lambda_grl) # <--- 旧的 CNN-LSTM 调用
        transfer_model = create_transfer_model(input_dim, num_classes, lambda_grl=lambda_grl)  # <--- 新的 MLP-DA 调用
        transfer_model.summary()
        print("✅ 迁移模型构建完成。")

        # --- 关键修改：划分源域训练/验证集 ---
        print("  - 划分源域训练集和验证集...")
        # X_s_train, X_s_val, y_s_train, y_s_val, y_s_train_cat, y_s_val_cat = train_test_split(
        #     X_source_cnn, y_source, y_source_cat, test_size=0.1, random_state=42, stratify=y_source) # <--- 旧的 CNN-LSTM 数据
        X_s_train, X_s_val, y_s_train, y_s_val, y_s_train_cat, y_s_val_cat = train_test_split(
            X_source_scaled, y_source, y_source_cat, test_size=0.1, random_state=42,
            stratify=y_source)  # <--- 新的 MLP-DA 数据
        # X_t_train = X_target_cnn  # 目标域数据全部用于训练 # <--- 旧的 CNN-LSTM 数据
        X_t_train = X_target_scaled  # 目标域数据全部用于训练 # <--- 新的 MLP-DA 数据
        print(f"    - 源域训练集: {X_s_train.shape}, 验证集: {X_s_val.shape}")
        print(f"    - 目标域训练集: {X_t_train.shape}")

        # --- 关键修改：使用自定义训练循环 ---
        epochs = 20
        batch_size = 64
        train_da_model(transfer_model, X_s_train, y_s_train_cat, X_t_train, X_s_val, y_s_val_cat, epochs, batch_size)
        print("✅ 对抗训练完成。")

        # --- 阶段五：目标域预测与标定 ---
        print("\n--- 阶段五：目标域预测与标定 ---")
        # final_model = Model(inputs=transfer_model.input, outputs=transfer_model.get_layer('class_output').output) # <--- 可以简化
        # --- 修改：直接使用主输出进行预测 ---
        y_target_pred_proba, _ = transfer_model.predict(X_target_scaled)  # <--- 直接获取分类输出
        y_target_pred_int = np.argmax(y_target_pred_proba, axis=1)
        y_target_pred_labels = le.inverse_transform(y_target_pred_int)
        print("✅ 目标域数据预测完成。")

        # --- 阶段六：迁移结果可视化展示与分析 ---
        print("\n--- 阶段六：迁移结果可视化展示与分析 ---")
        unique_labels, counts = np.unique(y_target_pred_labels, return_counts=True)
        plt.figure(figsize=(8, 6))
        colors = sns.color_palette("husl", len(unique_labels))
        plt.pie(counts, labels=unique_labels, autopct='%1.1f%%', startangle=140, colors=colors)
        # --- 修改：更新图表标题 ---
        plt.title('目标域预测结果类别分布 (MLP-DA 迁移)', fontsize=14, weight='bold')  # <--- 修改这里
        plt.axis('equal')
        # --- 修改：更新保存路径 ---
        save_path_pie = os.path.join(output_dir, '21_target_prediction_distribution_mlp_da.png')  # <--- 修改这里
        plt.savefig(save_path_pie, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 预测标签分布饼图已保存至: {save_path_pie}")

        max_probs = np.max(y_target_pred_proba, axis=1)
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.xlabel('预测置信度 (最大类概率)', fontsize=12)
        plt.ylabel('样本数量', fontsize=12)
        # --- 修改：更新图表标题 ---
        plt.title('目标域预测置信度分布 (MLP-DA 迁移)', fontsize=14, weight='bold')  # <--- 修改这里
        plt.grid(True, alpha=0.3)
        mean_conf = np.mean(max_probs)
        median_conf = np.median(max_probs)
        stats_text = f'均值: {mean_conf:.3f}\n中位数: {median_conf:.3f}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        # --- 修改：更新保存路径 ---
        save_path_hist = os.path.join(output_dir, '21_prediction_confidence_histogram_mlp_da.png')  # <--- 修改这里
        plt.savefig(save_path_hist, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 预测置信度直方图已保存至: {save_path_hist}")
        print(f"📊 预测置信度统计 - 均值: {mean_conf:.4f}, 中位数: {median_conf:.4f}")

        results_df = pd.DataFrame({
            'filename': target_filenames,
            'predicted_label': y_target_pred_labels,
            'confidence': max_probs
        })
        results_df = results_df.sort_values(by='filename').reset_index(drop=True)
        # --- 修改：更新保存路径 ---
        RESULTS_CSV_PATH = os.path.join(output_dir, '21_target_domain_predictions_mlp_da.csv')  # <--- 修改这里
        results_df.to_csv(RESULTS_CSV_PATH, index=False)
        print(f"✅ 目标域预测结果已保存至: {RESULTS_CSV_PATH}")

        # --- 修改：更新最终打印信息 ---
        print(f"\n🏆 任务三迁移诊断完成 (MLP-DA + 领域自适应)！")  # <--- 修改这里
        print(f"   - 使用的基础模型: MLP-DA (选自任务二，输入为聚合特征)")  # <--- 修改这里
        print(f"   - 迁移策略: 领域自适应 (Domain Adaptation with GRL + 自定义训练循环)")  # <--- 保持不变或微调
        print(f"   - 预测结果已保存在: {RESULTS_CSV_PATH}")
        print(f"   - 可视化图表已保存在: {os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"❌ 在模型构建、训练或预测阶段发生错误: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
