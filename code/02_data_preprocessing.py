import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample


# ==============================================================================
# 1. 文件读取工具 (与之前脚本一致)
# ==============================================================================
def load_mat_file(file_path: str):
    """
    读取 .mat 文件，能自动处理 v7.3 和旧版本格式。
    """
    try:
        return scipy.io.loadmat(file_path)
    except NotImplementedError:
        import h5py
        mat_data = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                mat_data[key] = np.array(f[key])
        return mat_data
    except Exception as e:
        print(f"读取文件 {os.path.basename(file_path)} 时发生错误: {e}")
        return None


# ==============================================================================
# 2. 数据加载、预处理与分段核心函数
# ==============================================================================
def load_preprocess_and_segment(df_selected_files, target_sr=32000, segment_len=4096, stride=512):
    """
    对筛选出的文件进行加载、重采样和重叠分段。

    参数:
    df_selected_files: 包含待处理文件信息的DataFrame。
    target_sr: 目标采样率 (Hz)，统一到目标域的32kHz。
    segment_len: 每个样本段的长度 (点数)。
    stride: 滑动窗口的步长，控制重叠度。步长越小，样本越多。
    """
    all_segments = []
    all_labels = []
    all_rpms = []

    print("🚀 开始进行数据预处理（重采样和分段）...")

    for index, row in df_selected_files.iterrows():
        file_path = row['file_path']
        label = row['fault_type']

        # 1. 加载 .mat 文件
        mat_data = load_mat_file(file_path)
        if not mat_data:
            continue

        # 2. 提取DE传感器信号和RPM
        de_key = next((key for key in mat_data if key.endswith('DE_time')), None)
        rpm_key = next((key for key in mat_data if key.endswith('RPM')), None)

        if not de_key or not rpm_key:
            print(f"  - 警告：文件 {os.path.basename(file_path)} 中缺少DE信号或RPM信息，已跳过。")
            continue

        signal = mat_data[de_key].flatten()
        rpm = mat_data[rpm_key].flatten()[0]

        # 3. 重采样
        original_sr_str = row['sampling_rate']
        original_sr = 48000 if original_sr_str == '48k' else 12000

        if original_sr != target_sr:
            num_samples = int(len(signal) * target_sr / original_sr)
            signal = resample(signal, num_samples)

        # 4. 信号分段 (使用重叠滑动窗口)
        num_generated = 0
        for i in range(0, len(signal) - segment_len + 1, stride):
            segment = signal[i: i + segment_len]
            all_segments.append(segment)
            all_labels.append(label)
            all_rpms.append(rpm)  # 为每个样本段记录其来源的RPM
            num_generated += 1

        print(f"  - ✅ 已处理: {row['filename']} (RPM: {rpm}) -> 生成了 {num_generated} 个样本段。")

    print("\n✅ 数据预处理完成！")
    return np.array(all_segments), np.array(all_labels), np.array(all_rpms)


# ==============================================================================
# 3. 主程序
# ==============================================================================
if __name__ == "__main__":
    # --- 配置参数 ---
    TARGET_SAMPLE_RATE = 32000  # 目标采样率 (Hz)
    SEGMENT_LENGTH = 4096  # 每个样本段的长度 (点数)
    STRIDE = 512  # 滑动窗口的步长 (点数)

    # --- 输入路径 ---
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    SELECTED_FILES_CSV = os.path.join(PROCESSED_DIR, 'step1_selected_source_files.csv')

    # --- 输出路径 ---
    SEGMENTS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'source_segments.npy')
    LABELS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'source_labels.npy')
    RPMS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'source_rpms.npy')

    # 1. 加载筛选文件列表
    if not os.path.exists(SELECTED_FILES_CSV):
        print(f"‼️ 错误：找不到筛选结果文件 {SELECTED_FILES_CSV}。请先运行 01_data_selection.py 脚本。")
    else:
        df_selected = pd.read_csv(SELECTED_FILES_CSV)
        print(f"成功加载筛选文件列表，共 {len(df_selected)} 个文件。")

        # 2. 执行预处理和分段
        segments, labels, rpms = load_preprocess_and_segment(
            df_selected_files=df_selected,
            target_sr=TARGET_SAMPLE_RATE,
            segment_len=SEGMENT_LENGTH,
            stride=STRIDE
        )

        print(f"\n📊 预处理结果统计:")
        print(f"  - 总样本段数量: {len(segments)}")
        if len(segments) > 0:
            print(f"  - 样本段长度: {segments.shape[1]}")
        print(f"  - 标签数量: {len(labels)}")

        # 3. 保存处理结果
        np.save(SEGMENTS_OUTPUT_PATH, segments)
        np.save(LABELS_OUTPUT_PATH, labels)
        np.save(RPMS_OUTPUT_PATH, rpms)

        print(f"\n💾 预处理后的样本段已保存至: {SEGMENTS_OUTPUT_PATH}")
        print(f"💾 对应的标签已保存至: {LABELS_OUTPUT_PATH}")
        print(f"💾 对应的转速已保存至: {RPMS_OUTPUT_PATH}")
        print("\n🎉 任务一（续）的数据预处理环节完成！我们现在拥有了丰富的、可用于特征提取的样本集。")