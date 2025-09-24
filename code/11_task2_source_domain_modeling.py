# 11_task2_source_domain_modeling.py
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier  # 添加随机森林
# --- 新增导入 ---
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
# --- 新增导入 ---
import lightgbm as lgb
from sklearn import svm
# --- 新增导入结束 ---
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

    # --- 修改：初始化所有模型的预测结果列表 ---
    all_y_test = []
    xgb_all_y_pred = []
    rf_all_y_pred = []
    svm_all_y_pred = [] # 新增
    lgb_all_y_pred = [] # 新增
    cnn_lstm_all_y_pred = []
    # --- 修改结束 ---

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_raw, y, groups)):
        print(f"\n--- 第 {fold + 1}/{N_SPLITS} 折 ---")
        X_train_raw, X_test_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        all_y_test.extend(y_test) # 收集测试标签

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
        # 新增：SVM模型
        # ======================================================================
        print("  - 正在训练 SVM...")
        # SVM 需要 probability=True 来获取预测概率
        svm_model = svm.SVC(
            kernel='rbf', # 常用核函数
            C=1.0,        # 正则化参数
            gamma='scale', # 核函数系数
            random_state=RANDOM_STATE,
            class_weight='balanced', # 处理类别不平衡
            probability=True # 启用概率预测
        )
        svm_model.fit(X_train, y_train)
        y_pred_svm = svm_model.predict(X_test)
        svm_all_y_pred.extend(y_pred_svm) # 收集SVM预测

        # ======================================================================
        # 新增：LightGBM模型
        # ======================================================================
        print("  - 正在训练 LightGBM (含早停)...")
        # 注意：LightGBM 使用 validation set 进行早停
        X_train_lgb_sub, X_val_lgb, y_train_lgb_sub, y_val_lgb = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
        )

        # 计算样本权重 (LightGBM 也可以通过参数处理，这里演示手动计算)
        # sample_weights_train_lgb = compute_balanced_sample_weights(y_train_lgb_sub)
        # sample_weights_val_lgb = compute_balanced_sample_weights(y_val_lgb)

        # LightGBM 参数 (参考 XGBoost)
        lgb_params = {
            'objective': 'multiclass',
            'num_class': NUM_CLASSES,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31, # 控制模型复杂度
            'learning_rate': 0.05,
            'feature_fraction': 0.9, # 类似 colsample_bytree
            'bagging_fraction': 0.8, # 类似 subsample
            'bagging_freq': 5,
            'verbose': -1, # 减少输出
            'random_state': RANDOM_STATE,
            # 'class_weight': 'balanced' # 也可以直接设置
        }

        # 创建 LightGBM Dataset
        train_data = lgb.Dataset(X_train_lgb_sub, label=y_train_lgb_sub) # , weight=sample_weights_train_lgb)
        val_data = lgb.Dataset(X_val_lgb, label=y_val_lgb, reference=train_data) # , weight=sample_weights_val_lgb)

        # 训练模型
        lgb_model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=0) # 禁止训练日志输出
            ]
        )

        # 预测
        y_pred_lgb_prob = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
        y_pred_lgb = np.argmax(y_pred_lgb_prob, axis=1)
        lgb_all_y_pred.extend(y_pred_lgb) # 收集LightGBM预测

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
    # --- 修改：更新模型结果字典 ---
    models_results = {
        "XGBoost": xgb_all_y_pred,
        "RandomForest": rf_all_y_pred,
        "SVM": svm_all_y_pred, # 新增
        "LightGBM": lgb_all_y_pred, # 新增
        "CNN-LSTM": cnn_lstm_all_y_pred
    }
    # --- 修改结束 ---

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

    # --- 新增：生成所有模型的混淆矩阵和精度对比图 ---
    print("\n🚀 步骤 3.5: 生成所有模型的性能对比图表...")
    try:
        # 1. 混淆矩阵对比 (子图)
        fig_cm, axes = plt.subplots(2, 3, figsize=(20, 12)) # 假设最多5个模型，用2x3布局
        axes = axes.ravel()

        for idx, (model_name, y_pred) in enumerate(models_results.items()):
            if idx < len(axes):
                cm = confusion_matrix(all_y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[idx])
                axes[idx].set_title(f'{model_name} 混淆矩阵', fontsize=14)
                axes[idx].set_xlabel('Predicted Label')
                axes[idx].set_ylabel('True Label')

        # 隐藏多余的子图
        for j in range(len(models_results), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        save_path_cm_all = os.path.join(OUTPUT_DIR, '任务二-所有模型混淆矩阵对比.png')
        plt.savefig(save_path_cm_all, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 所有模型混淆矩阵对比图已保存至: {save_path_cm_all}")

        # 2. 精度/指标对比 (条形图)
        comparison_data = []
        for model_name, y_pred in models_results.items():
            accuracy = accuracy_score(all_y_test, y_pred)
            report_dict = classification_report(all_y_test, y_pred, target_names=le.classes_, output_dict=True)
            macro_f1 = report_dict['macro avg']['f1-score']
            weighted_f1 = report_dict['weighted avg']['f1-score']
            ir_recall = report_dict.get('IR', {}).get('recall', 0)
            b_precision = report_dict.get('B', {}).get('precision', 0)
            n_recall = report_dict.get('N', {}).get('recall', 0) # 假设 N 类存在

            comparison_data.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Macro F1': macro_f1,
                'Weighted F1': weighted_f1,
                'IR Recall': ir_recall,
                'B Precision': b_precision,
                'N Recall': n_recall
            })

        df_comparison = pd.DataFrame(comparison_data)
        # melted_df = df_comparison.melt(id_vars=['Model'], var_name='Metric', value_name='Score')

        fig_metrics, ax = plt.subplots(figsize=(12, 8))
        # x_pos = np.arange(len(df_comparison))
        # width = 0.15 # 每个指标条形的宽度
        # metrics_to_plot = ['Accuracy', 'Macro F1', 'Weighted F1', 'IR Recall', 'B Precision']
        # for i, metric in enumerate(metrics_to_plot):
        #     ax.bar(x_pos + i*width, df_comparison[metric], width, label=metric)
        # ax.set_xlabel('Models')
        # ax.set_ylabel('Score')
        # ax.set_title('模型性能指标对比')
        # ax.set_xticks(x_pos + width * (len(metrics_to_plot)-1) / 2)
        # ax.set_xticklabels(df_comparison['Model'])
        # ax.legend()
        # plt.xticks(rotation=45)

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

        save_path_metrics = os.path.join(OUTPUT_DIR, '任务二-模型性能指标对比.png')
        plt.savefig(save_path_metrics, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 模型性能指标对比图已保存至: {save_path_metrics}")

    except Exception as e:
        print(f"⚠️ 生成性能对比图表时出错: {e}")
    # --- 新增结束 ---


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
    # 新增：SVM最终模型
    # ======================================================================
    print("  - 正在训练最终的SVM模型...")
    final_svm_model = svm.SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=RANDOM_STATE,
        class_weight='balanced',
        probability=True # 确保可以获取概率
    )
    final_svm_model.fit(X_scaled_full, y) # 使用全部数据训练

    # ======================================================================
    # 新增：LightGBM最终模型
    # ======================================================================
    print("  - 正在训练最终的LightGBM模型...")
    # LightGBM 参数 (可以微调)
    final_lgb_params = {
        'objective': 'multiclass',
        'num_class': NUM_CLASSES,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        # 'class_weight': 'balanced' # 可选
    }

    # 创建最终训练数据集
    final_train_data = lgb.Dataset(X_scaled_full, label=y) # , weight=sample_weights_full) # 如果需要权重

    # 训练最终模型 (不使用早停，或使用全部数据作为验证集进行少量轮次)
    final_lgb_model = lgb.train(
        final_lgb_params,
        final_train_data,
        num_boost_round=100 # 根据 CV 结果调整
    )


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
    # --- 新增：保存 SVM 和 LightGBM 模型 ---
    joblib.dump(final_svm_model, os.path.join(OUTPUT_DIR, 'final_svm_model.joblib')) # 保存SVM
    # LightGBM Booster 对象需要特殊保存
    final_lgb_model.save_model(os.path.join(OUTPUT_DIR, 'final_lgb_model.txt')) # 保存LightGBM
    # --- 新增结束 ---
    joblib.dump(final_scaler, os.path.join(OUTPUT_DIR, 'final_scaler.joblib'))
    joblib.dump(le, os.path.join(OUTPUT_DIR, 'final_label_encoder.joblib'))
    # --- 【核心修正】将 .keras 改为 .h5 以确保兼容性 ---
    # 修改文件扩展名为 .weights.h5
    final_cnn_lstm_model.save_weights(os.path.join(OUTPUT_DIR, 'final_cnn_lstm_model.weights.h5'))  # 修改为CNN+LSTM

    print(f"✅ 最终模型及预处理器已保存至: {os.path.abspath(OUTPUT_DIR)}")
    print("\n🎉 任务二：源域故障诊断（最终版）全部工作已圆满完成！")
