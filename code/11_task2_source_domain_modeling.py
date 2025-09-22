import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold, train_test_split  # 【核心】导入分组分层K折交叉验证
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, GlobalAveragePooling1D, \
    Dense, Dropout, multiply, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# 修正import语句
from sklearn.utils.class_weight import compute_class_weight  # 正确的函数名


# ==============================================================================
# 0. 辅助函数
# ==============================================================================
def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 已设置中文字体。")


def create_cnn_attention_model(input_shape, num_classes):
    """创建1D-CNN + Attention模型"""
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    cnn_out = ReLU()(x)  # Shape: (batch_size, steps, 128)

    # --- 【核心修正】在时间步维度上计算注意力 ---
    # Step 1: 生成每个时间步的"能量"分数
    attention = Dense(1, activation='tanh')(cnn_out)  # Shape: (batch_size, steps, 1)
    # Step 2: 在时间步（axis=1）上进行Softmax，得到归一化的注意力权重
    attention_probs = Activation('softmax', name='attention_weights')(attention)  # Shape: (batch_size, steps, 1)
    # Step 3: 将权重应用到原始特征上
    attention_mul = multiply([cnn_out, attention_probs])  # Shape: (batch_size, steps, 128)

    x = GlobalAveragePooling1D()(attention_mul)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax', name='main_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='cnn_attention_model')  # 只返回主输出
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


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

    all_y_test, xgb_all_y_pred, cnn_all_y_pred = [], [], []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_raw, y, groups)):
        print(f"\n--- 第 {fold + 1}/{N_SPLITS} 折 ---")
        X_train_raw, X_test_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        all_y_test.extend(y_test)

        print("  - 正在训练 XGBoost (含早停调参)...")
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
        )

        # 转换为 DMatrix 格式
        dtrain = xgb.DMatrix(X_train_sub, label=y_train_sub)
        dval = xgb.DMatrix(X_val, label=y_val)

        # 设置参数
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': NUM_CLASSES,  # 类别数
            'random_state': RANDOM_STATE
        }

        # 使用原生训练 API + 早停
        evals_result = {}
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=15,
            evals_result=evals_result,
            verbose_eval=False
        )

        print(f"    最佳树数量: {bst.best_iteration}")

        # 使用 DMatrix 进行预测
        dtest = xgb.DMatrix(X_test)
        class_probs = bst.predict(dtest)
        y_pred_xgb = np.argmax(class_probs, axis=1)
        xgb_all_y_pred.extend(y_pred_xgb)

        print("  - 正在训练 1D-CNN + Attention (含早停)...")
        X_train_cnn, X_test_cnn = np.expand_dims(X_train, axis=2), np.expand_dims(X_test, axis=2)
        y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
        cnn_model = create_cnn_attention_model(input_shape=(X_train_cnn.shape[1], 1), num_classes=NUM_CLASSES)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

        # 修正：计算类别权重并正确使用
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        # 修正：CNN模型训练时正确使用标签格式和类别权重
        cnn_model.fit(X_train_cnn, y_train_cat,  # 修正：使用正确的标签格式
                      epochs=100, batch_size=64, verbose=0,
                      validation_split=0.2,
                      # 修正：添加类别权重
                      class_weight=class_weight_dict,
                      callbacks=[early_stopping])
        y_pred_cnn_prob = cnn_model.predict(X_test_cnn)
        y_pred_cnn = np.argmax(y_pred_cnn_prob, axis=1)
        cnn_all_y_pred.extend(y_pred_cnn)

    print("\n🚀 步骤 3: 交叉验证完成，汇总评估结果...")
    models_results = {"XGBoost": xgb_all_y_pred, "1D-CNN-Attention": cnn_all_y_pred}
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

    print("  - 正在训练最终的XGBoost模型...")
    final_xgb_model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=RANDOM_STATE,
                                        n_estimators=200, scale_pos_weight=1)  # 使用一个合理的树数量
    final_xgb_model.fit(X_scaled_full, y)

    print("  - 正在训练最终的1D-CNN+Attention模型...")
    X_cnn_full = np.expand_dims(X_scaled_full, axis=2)
    y_cat_full = to_categorical(y, num_classes=NUM_CLASSES)
    final_cnn_model = create_cnn_attention_model(input_shape=(X_cnn_full.shape[1], 1), num_classes=NUM_CLASSES)
    # --- 【核心修正】为最终模型训练添加早停，防止过拟合 ---
    early_stopping_final = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)

    # 为最终模型也添加类别权重
    class_weights_final = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict_final = dict(enumerate(class_weights_final))

    final_cnn_model.fit(X_cnn_full, y_cat_full, epochs=100, batch_size=64, verbose=0,
                        class_weight=class_weight_dict_final,  # 添加类别权重
                        callbacks=[early_stopping_final])

    print("\n🚀 步骤 5: 正在保存最终产出...")
    joblib.dump(final_xgb_model, os.path.join(OUTPUT_DIR, 'final_xgb_model.joblib'))
    joblib.dump(final_scaler, os.path.join(OUTPUT_DIR, 'final_scaler.joblib'))
    joblib.dump(le, os.path.join(OUTPUT_DIR, 'final_label_encoder.joblib'))
    # --- 【核心修正】将 .keras 改为 .h5 以确保兼容性 ---
    final_cnn_model.save(os.path.join(OUTPUT_DIR, 'final_cnn_model.h5'))

    print(f"✅ 最终模型及预处理器已保存至: {os.path.abspath(OUTPUT_DIR)}")
    print("\n🎉 任务二：源域故障诊断（最终版）全部工作已圆满完成！")