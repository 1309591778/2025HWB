# 11_task2_source_domain_modeling.py
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


# ==============================================================================
# 0. 辅助函数
# ==============================================================================
def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")


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
        self.lambda_val = lambda_val

    def call(self, x):
        return grad_reverse(x, self.lambda_val)

    def get_config(self):
        config = super(GradReverse, self).get_config()
        config.update({'lambda_val': self.lambda_val})
        return config


# --- 新增结束 ---


# --- 修改：定义包含领域自适应的简化 MLP 模型 ---
def create_mlp_da_model(input_dim, num_classes, lambda_grl=1.0):
    """
    创建用于源域训练的简化 MLP 模型，包含领域自适应组件。
    """
    # 1. 输入层
    inputs = Input(shape=(input_dim,))

    # 2. 特征提取器 (MLP)
    shared = Dense(128, activation='relu', name='feature_extractor_1')(inputs)
    shared = Dropout(0.5)(shared)
    shared = Dense(64, activation='relu', name='feature_extractor_2')(shared)
    shared = Dropout(0.5)(shared)
    features_before_grl = Dense(32, activation='relu', name='feature_extractor_3')(shared)

    # --- 新增：领域自适应分支 ---
    # 3a. 梯度反转层 (GRL)
    grl = GradReverse(lambda_val=lambda_grl)(features_before_grl)

    # 3b. 领域判别器 (Domain Discriminator)
    d_net = Dense(32, activation='relu')(grl)
    d_net = Dropout(0.5)(d_net)
    domain_output = Dense(1, activation='sigmoid', name='domain_output')(d_net)
    # --- 新增结束 ---

    # 4. 主任务分类头
    c_net = Dense(64, activation='relu')(features_before_grl)
    c_net = Dropout(0.5)(c_net)
    class_output = Dense(num_classes, activation='softmax', name='class_output')(c_net)

    # 5. 构建模型
    model = Model(inputs=inputs, outputs=[class_output, domain_output])

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'class_output': 'categorical_crossentropy',
            'domain_output': 'binary_crossentropy'
        },
        loss_weights={
            'class_output': 1.0,
            'domain_output': 1.0  # 可以调整这个权重 lambda_domain
        },
        metrics={
            'class_output': 'accuracy',
            'domain_output': 'accuracy'
        }
    )

    return model


# --- 修改结束 ---


def compute_balanced_sample_weights(y, n_class_index=2, n_class_weight=3.0):
    """计算平衡的样本权重，给N类更高的权重"""
    n_classes = len(np.unique(y))
    class_weights = {i: 1.0 for i in range(n_classes)}
    class_weights[n_class_index] = n_class_weight
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
        'num_class': 4,
        'random_state': 42,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.2,
    }
    return params


def train_xgb_model_cv(X_train, y_train, X_val, y_val, sample_weights_train=None, sample_weights_val=None):
    """训练XGBoost模型（交叉验证用）"""
    if sample_weights_train is not None and sample_weights_val is not None:
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights_train)
        dval = xgb.DMatrix(X_val, label=y_val, weight=sample_weights_val)
    else:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

    params = create_xgb_model_params()
    params['num_class'] = len(np.unique(y_train))

    evals_result = {}
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=30,
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
    NUM_CLASSES = len(le.classes_)
    input_dim = X_raw.shape[1]  # 获取特征维度

    print(f"\n🚀 步骤 2: 开始 {N_SPLITS} 折分组分层交叉验证...")
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    all_y_test, xgb_all_y_pred, rf_all_y_pred, mlp_da_all_y_pred = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_raw, y, groups)):
        print(f"\n--- 第 {fold + 1}/{N_SPLITS} 折 ---")
        X_train_raw, X_test_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        all_y_test.extend(y_test)

        # ======================================================================
        # XGBoost模型保持不变
        # ======================================================================
        print("  - 正在训练 XGBoost (含早停调参)...")
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
        )

        sample_weights_train = compute_balanced_sample_weights(y_train_sub)
        sample_weights_val = compute_balanced_sample_weights(y_val)

        bst = train_xgb_model_cv(
            X_train_sub, y_train_sub, X_val, y_val,
            sample_weights_train, sample_weights_val
        )

        print(f"    最佳树数量: {bst.best_iteration}")

        y_pred_xgb, class_probs = predict_xgb_model(bst, X_test)
        xgb_all_y_pred.extend(y_pred_xgb)

        # ======================================================================
        # 随机森林模型保持不变
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
        # 修改：简化 MLP + DA 模型 (替代 CNN-LSTM)
        # ======================================================================
        print("  - 正在训练 简化 MLP + DA 模型 (含早停)...")
        y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
        # 注意：这里不再需要 expand_dims，直接使用 (batch_size, features)
        # 创建模型
        mlp_da_model = create_mlp_da_model(input_dim=input_dim, num_classes=NUM_CLASSES, lambda_grl=1.0)
        early_stopping = EarlyStopping(monitor='val_class_output_accuracy', mode='max', patience=20,
                                       restore_best_weights=True)

        # 准备伪领域标签 (在源域训练中，所有样本都标记为源域 0)
        domain_labels_source_train = np.zeros((X_train.shape[0], 1))
        domain_labels_source_val = np.zeros((X_train_sub.shape[0], 1))  # 用于验证集

        # 训练模型 (简化处理，实际应用中可使用自定义循环进行更精确的DA)
        # 这里我们简化处理，只在训练集上训练主任务和领域任务
        # 验证集也标记为源域
        X_train_sub_scaled = scaler.transform(X_train_sub)  # 验证集也需要标准化
        history = mlp_da_model.fit(
            X_train,
            {"class_output": y_train_cat, "domain_output": domain_labels_source_train},
            epochs=100,
            batch_size=64,
            verbose=0,
            validation_data=(
                X_train_sub_scaled,
                {"class_output": to_categorical(y_train_sub, num_classes=NUM_CLASSES),
                 "domain_output": domain_labels_source_val}
            ),
            callbacks=[early_stopping]
        )
        print(f"    训练完成，最佳Epoch: {len(history.history['loss']) - early_stopping.patience}")

        # 预测 (只使用分类输出)
        y_pred_mlp_da_prob, _ = mlp_da_model.predict(X_test)
        y_pred_mlp_da = np.argmax(y_pred_mlp_da_prob, axis=1)
        mlp_da_all_y_pred.extend(y_pred_mlp_da)

    print("\n🚀 步骤 3: 交叉验证完成，汇总评估结果...")
    # 更新模型结果字典
    models_results = {
        "XGBoost": xgb_all_y_pred,
        "RandomForest": rf_all_y_pred,
        "MLP-DA": mlp_da_all_y_pred  # 更新键名
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
    sample_weights_full = compute_balanced_sample_weights(y)

    final_xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.2,
    )
    final_xgb_model.fit(X_scaled_full, y, sample_weight=sample_weights_full)

    # ======================================================================
    # 随机森林最终模型保持不变
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
    # 修改：训练最终的 MLP + DA 模型 (替代 CNN-LSTM)
    # ======================================================================
    print("  - 正在训练最终的 MLP + DA 模型...")
    y_cat_full = to_categorical(y, num_classes=NUM_CLASSES)
    # 创建最终模型
    final_mlp_da_model = create_mlp_da_model(input_dim=input_dim, num_classes=NUM_CLASSES, lambda_grl=1.0)
    early_stopping_final = EarlyStopping(monitor='class_output_accuracy', mode='max', patience=20,
                                         restore_best_weights=True)

    # 准备伪领域标签
    domain_labels_full = np.zeros((X_scaled_full.shape[0], 1))

    # 训练最终模型
    final_mlp_da_model.fit(
        X_scaled_full,
        {"class_output": y_cat_full, "domain_output": domain_labels_full},
        epochs=100,
        batch_size=64,
        verbose=0,
        callbacks=[early_stopping_final]
    )

    print("\n🚀 步骤 5: 正在保存最终产出...")
    joblib.dump(final_xgb_model, os.path.join(OUTPUT_DIR, 'final_xgb_model.joblib'))
    joblib.dump(final_rf_model, os.path.join(OUTPUT_DIR, 'final_rf_model.joblib'))
    joblib.dump(final_scaler, os.path.join(OUTPUT_DIR, 'final_scaler.joblib'))
    joblib.dump(le, os.path.join(OUTPUT_DIR, 'final_label_encoder.joblib'))

    # --- 修改：保存 MLP + DA 模型的权重 ---
    # 保存权重，使用 .weights.h5 扩展名
    final_mlp_da_model.save_weights(os.path.join(OUTPUT_DIR, 'final_mlp_da_model.weights.h5'))
    print("✅ MLP-DA 模型权重已保存。")
    # --- 修改结束 ---

    print(f"✅ 最终模型及预处理器已保存至: {os.path.abspath(OUTPUT_DIR)}")
    print("\n🎉 任务二：源域故障诊断（最终版）全部工作已圆满完成！")
