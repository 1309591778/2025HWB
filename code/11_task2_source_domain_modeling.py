import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier  # 添加随机森林
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, LSTM, \
    GlobalAveragePooling1D, Dense, Dropout, multiply, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# 修正import语句
from sklearn.utils.class_weight import compute_class_weight  # 正确的函数名
# 在现有导入后添加
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==============================================================================
# 0. 辅助函数
# ==============================================================================
def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")


def categorical_focal_loss(gamma=2., alpha=None): # <--- alpha 可以是列表或 None
    """
    Focal Loss for addressing class imbalance in categorical classification.
    Supports per-class alpha weights.
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)

        # --- 修改点：支持 alpha 为列表 ---
        if alpha is not None:
            # 假设 alpha 是一个列表或 numpy array，长度等于类别数
            # y_true 是 one-hot 编码，shape [batch_size, num_classes]
            # alpha 需要 reshape 成 [1, num_classes] 以便广播
            alpha_tensor = tf.constant(alpha, dtype=tf.float32)
            alpha_tensor = tf.reshape(alpha_tensor, [1, -1])
            # weight = alpha * y_true * (1 - y_pred)^gamma
            weight = alpha_tensor * y_true * tf.keras.backend.pow((1 - y_pred), gamma)
        else:
            # 如果 alpha 为 None，则不使用权重
            weight = y_true * tf.keras.backend.pow((1 - y_pred), gamma)
        # --- 修改结束 ---

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

    # 使用Focal Loss替代标准交叉熵损失
    num_classes = len(le.classes_)  # 通常是 4
    # 创建 alpha 列表，索引对应类别
    # 例如，给 B 类 (索引 0) 更高的权重
    alpha_list = [0.25] * num_classes  # 默认所有类权重 0.25
    b_class_index = 0  # 确认 B 类索引
    alpha_list[b_class_index] = 0.75  # 给 B 类更高的权重 (例如 0.75)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=categorical_focal_loss(gamma=2., alpha=alpha_list),  # <--- 使用 alpha 列表
        metrics=['accuracy']
    )
    return model


def compute_balanced_sample_weights(y, n_class_index=2, n_class_weight=3.0):
    """计算平衡的样本权重，给N类更高的权重"""
    # 获取类别数量
    n_classes = len(np.unique(y))

    # 默认权重为1.0
    class_weights = {i: 1.0 for i in range(n_classes)}

    # 给N类（假设索引为2）更高的权重
    class_weights[n_class_index] = n_class_weight

    # 为每个样本分配权重
    sample_weights = np.array([class_weights[label] for label in y])
    return sample_weights


# ==============================================================================
# 新增：XGBoost模型相关函数
# ==============================================================================
def create_xgb_model_params():
    """创建XGBoost模型参数"""
    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 4,  # 假设有4个类别
        'random_state': 42,
        # 优化参数
        'max_depth': 6,          # 降低深度避免过拟合
        'learning_rate': 0.1,    # 提高学习率
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,   # 增加最小权重
        'gamma': 0.2,            # 增加最小损失减少量
    }
    return params


def train_xgb_model_cv(X_train, y_train, X_val, y_val, sample_weights_train=None, sample_weights_val=None):
    """训练XGBoost模型（交叉验证用）"""
    # 转换为 DMatrix 格式
    if sample_weights_train is not None and sample_weights_val is not None:
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights_train)
        dval = xgb.DMatrix(X_val, label=y_val, weight=sample_weights_val)
    else:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

    # 获取参数
    params = create_xgb_model_params()
    params['num_class'] = len(np.unique(y_train))  # 动态设置类别数

    # 使用原生训练 API + 早停
    evals_result = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,     # 减少迭代次数
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=30, # 早停轮数
        evals_result=evals_result,
        verbose_eval=False
    )

    return bst


def predict_xgb_model(model, X_test):
    """使用XGBoost模型进行预测"""
    dtest = xgb.DMatrix(X_test)
    class_probs = model.predict(dtest)
    y_pred = np.argmax(class_probs, axis=1)
    return y_pred, class_probs

# ==============================================================================
# 1. 主程序
# ==============================================================================
if __name__ == "__main__":
    set_chinese_font()

    RANDOM_STATE = 42
    N_SPLITS = 5
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    FEATURES_PATH = os.path.join(PROCESSED_DIR, 'source_features_selected.csv')
    OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'task2_outputs_final')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🚀 步骤 1: 加载数据...")
    df_features = pd.read_csv(FEATURES_PATH)
    X_raw = df_features.drop(columns=['label', 'rpm', 'filename'])
    y_str = df_features['label']
    groups = df_features['filename']
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    # --- 【核心修正】定义全局常量 ---
    NUM_CLASSES = len(le.classes_)

    print(f"\n🚀 步骤 2: 开始 {N_SPLITS} 折分组分层交叉验证...")
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    all_y_test, xgb_all_y_pred, rf_all_y_pred, cnn_lstm_all_y_pred = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_raw, y, groups)):
        print(f"\n--- 第 {fold + 1}/{N_SPLITS} 折 ---")
        X_train_raw, X_test_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        all_y_test.extend(y_test)

        # ======================================================================
        # XGBoost模型保持不变（你要求的）
        # ======================================================================
        print("  - 正在训练 XGBoost (含早停调参)...")
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
        )

        # 计算样本权重
        sample_weights_train = compute_balanced_sample_weights(y_train_sub)
        sample_weights_val = compute_balanced_sample_weights(y_val)

        # 使用封装的函数训练XGBoost模型
        bst = train_xgb_model_cv(
            X_train_sub, y_train_sub, X_val, y_val,
            sample_weights_train, sample_weights_val
        )

        print(f"    最佳树数量: {bst.best_iteration}")

        # 使用封装的函数进行预测
        y_pred_xgb, class_probs = predict_xgb_model(bst, X_test)
        xgb_all_y_pred.extend(y_pred_xgb)

        # ======================================================================
        # 新增：随机森林模型
        # ======================================================================
        print("  - 正在训练随机森林...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        rf_all_y_pred.extend(y_pred_rf)

        # ======================================================================
        # 新增：CNN+LSTM模型
        # ======================================================================
        print("  - 正在训练 CNN+LSTM (含早停)...")
        X_train_cnn, X_test_cnn = np.expand_dims(X_train, axis=2), np.expand_dims(X_test, axis=2)
        y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
        cnn_lstm_model = create_cnn_lstm_model(input_shape=(X_train_cnn.shape[1], 1), num_classes=NUM_CLASSES)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

        # 计算类别权重（添加这部分）
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        # 特别加强B类和N类的权重
        # 假设B类索引为0，N类索引为2（需要根据实际情况调整）
        b_class_index = 0  # 根据你的标签编码调整
        n_class_index = 2  # 根据你的标签编码调整

        if b_class_index in class_weight_dict:
            class_weight_dict[b_class_index] = class_weight_dict[b_class_index] * 2.0  # B类权重加倍
        if n_class_index in class_weight_dict:
            class_weight_dict[n_class_index] = class_weight_dict[n_class_index] * 1.5  # N类权重增加50%

        cnn_lstm_model.fit(
            X_train_cnn, y_train_cat,
            epochs=100,  # 减少训练轮数
            batch_size=64,  # 增大批次大小节省显存
            verbose=0,
            validation_split=0.2,
            class_weight=class_weight_dict,  # 添加类别权重
            callbacks=[early_stopping]
        )
        y_pred_cnn_lstm_prob = cnn_lstm_model.predict(X_test_cnn)
        y_pred_cnn_lstm = np.argmax(y_pred_cnn_lstm_prob, axis=1)
        cnn_lstm_all_y_pred.extend(y_pred_cnn_lstm)

    print("\n🚀 步骤 3: 交叉验证完成，汇总评估结果...")
    models_results = {
        "XGBoost": xgb_all_y_pred,
        "RandomForest": rf_all_y_pred,
        "CNN-LSTM": cnn_lstm_all_y_pred
    }

    for model_name, y_pred in models_results.items():
        accuracy = accuracy_score(all_y_test, y_pred)
        report_dict = classification_report(all_y_test, y_pred, target_names=le.classes_, output_dict=True)
        print(f"\n========== {model_name} 最终性能评估 ==========")
        print(f"📊 总体准确率 (5折交叉验证): {accuracy:.4f}")
        print("\n📊 分类报告:")
        print(classification_report(all_y_test, y_pred, target_names=le.classes_))

        print("\n--- 关键风险指标分析 ---")
        ir_recall = report_dict.get('IR', {}).get('recall', 'N/A')
        b_precision = report_dict.get('B', {}).get('precision', 'N/A')
        print(f"  - 内圈故障(IR)的召回率 (避免漏判): {ir_recall:.4f}")
        print(f"  - 滚动体故障(B)的精确率 (避免误判): {b_precision:.4f}")

        cm = confusion_matrix(all_y_test, y_pred)
        np.fill_diagonal(cm, 0)
        if np.max(cm) > 0:
            misclass_idx = np.unravel_index(np.argmax(cm), cm.shape)
            true_label, pred_label = le.classes_[misclass_idx[0]], le.classes_[misclass_idx[1]]
            print(f"  - 最主要的混淆: 模型倾向于将真实的'{true_label}'类别误判为'{pred_label}'类别 ({np.max(cm)}次)。")

        cm_full = confusion_matrix(all_y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'{model_name} 模型混淆矩阵 (5折交叉验证)', fontsize=16)
        plt.savefig(os.path.join(OUTPUT_DIR, f'任务二-{model_name}-混淆矩阵.png'), dpi=300)
        plt.close()
        print(f"✅ {model_name} 的混淆矩阵已保存。")

    print("\n🚀 步骤 4: 正在使用全部源域数据训练最终模型以用于任务三...")
    final_scaler = StandardScaler().fit(X_raw)
    X_scaled_full = final_scaler.transform(X_raw)

    # ======================================================================
    # XGBoost最终模型保持不变
    # ======================================================================
    print("  - 正在训练最终的XGBoost模型...")
    # 计算样本权重，重点提升N类权重
    sample_weights_full = compute_balanced_sample_weights(y)

    final_xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_estimators=500,  # 增加树数量
        max_depth=6,  # 降低深度避免过拟合
        learning_rate=0.1,  # 合适的学习率
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.2,
    )
    final_xgb_model.fit(X_scaled_full, y, sample_weight=sample_weights_full)

    # ======================================================================
    # 新增：随机森林最终模型
    # ======================================================================
    print("  - 正在训练最终的随机森林模型...")
    final_rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    final_rf_model.fit(X_scaled_full, y)

    # ======================================================================
    # 新增：CNN+LSTM最终模型
    # ======================================================================
    print("  - 正在训练最终的CNN+LSTM模型...")
    X_cnn_full = np.expand_dims(X_scaled_full, axis=2)
    y_cat_full = to_categorical(y, num_classes=NUM_CLASSES)
    final_cnn_lstm_model = create_cnn_lstm_model(input_shape=(X_cnn_full.shape[1], 1), num_classes=NUM_CLASSES)
    # --- 【核心修正】为最终模型训练添加早停，防止过拟合 ---
    early_stopping_final = EarlyStopping(monitor='accuracy', patience=20, restore_best_weights=True)

    # 为最终模型也添加类别权重（添加这部分）
    class_weights_final = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict_final = dict(enumerate(class_weights_final))

    # 特别加强B类和N类的权重
    b_class_index = 0  # 根据你的标签编码调整
    n_class_index = 2  # 根据你的标签编码调整

    if b_class_index in class_weight_dict_final:
        class_weight_dict_final[b_class_index] = class_weight_dict_final[b_class_index] * 3.0
    if n_class_index in class_weight_dict_final:
        class_weight_dict_final[n_class_index] = class_weight_dict_final[n_class_index] * 1.5

    final_cnn_lstm_model.fit(X_cnn_full, y_cat_full, epochs=100, batch_size=64, verbose=0,
                             class_weight=class_weight_dict_final,  # 添加类别权重
                             callbacks=[early_stopping_final])

    print("\n🚀 步骤 5: 正在保存最终产出...")
    joblib.dump(final_xgb_model, os.path.join(OUTPUT_DIR, 'final_xgb_model.joblib'))
    joblib.dump(final_rf_model, os.path.join(OUTPUT_DIR, 'final_rf_model.joblib'))  # 新增
    joblib.dump(final_scaler, os.path.join(OUTPUT_DIR, 'final_scaler.joblib'))
    joblib.dump(le, os.path.join(OUTPUT_DIR, 'final_label_encoder.joblib'))
    # --- 【核心修正】将 .keras 改为 .h5 以确保兼容性 ---
    # 修改文件扩展名为 .weights.h5
    final_cnn_lstm_model.save_weights(os.path.join(OUTPUT_DIR, 'final_cnn_lstm_model.weights.h5'))
    print("✅ CNN-LSTM 模型权重已保存。")

    print(f"✅ 最终模型及预处理器已保存至: {os.path.abspath(OUTPUT_DIR)}")
    print("\n🎉 任务二：源域故障诊断（最终版）全部工作已圆满完成！")