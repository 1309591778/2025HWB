import os
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample


# ==============================================================================
# 1. 工具函数 (从之前的脚本复用)
# ==============================================================================
def load_mat_file(file_path: str):
    """读取 .mat 文件"""
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
# 2. 数据加载、预处理与分段核心函数 (核心修改)
# ==============================================================================
def load_preprocess_and_segment(df_selected_files, target_sr=32000, segment_len=3200, stride=400):
    """
    【最终版】对筛选出的文件进行加载、重采样和重叠分段，并记录每个分段的来源文件名。
    """
    all_segments = []
    all_labels = []
    all_rpms = []
    all_filenames = []  # <--- 【新增】用于存储文件名的列表

    print("🚀 开始进行数据预处理（重采样和分段）...")

    for index, row in df_selected_files.iterrows():
        file_path = row['file_path']
        label = row['fault_type']

        mat_data = load_mat_file(file_path)
        if not mat_data:
            continue

        de_key = next((key for key in mat_data if key.endswith('DE_time')), None)
        rpm_key = next((key for key in mat_data if key.endswith('RPM')), None)

        if not de_key or not rpm_key:
            print(f"  - 警告：文件 {os.path.basename(file_path)} 中缺少DE信号或RPM信息，已跳过。")
            continue

        signal = mat_data[de_key].flatten()
        rpm = mat_data[rpm_key].flatten()[0]

        original_sr_str = row['sampling_rate']
        original_sr = 48000 if original_sr_str == '48k' else 12000

        if original_sr != target_sr:
            num_samples = int(len(signal) * target_sr / original_sr)
            signal = resample(signal, num_samples)

        num_generated = 0
        for i in range(0, len(signal) - segment_len + 1, stride):
            segment = signal[i: i + segment_len]
            all_segments.append(segment)
            all_labels.append(label)
            all_rpms.append(rpm)
            all_filenames.append(row['filename'])  # <--- 【新增】为每个样本段记录其来源文件名
            num_generated += 1

        print(f"  - ✅ 已处理: {row['filename']} (RPM: {rpm}) -> 生成了 {num_generated} 个样本段。")

    print("\n✅ 数据预处理完成！")
    # 【修改】返回新增的文件名数组
    return np.array(all_segments), np.array(all_labels), np.array(all_rpms), np.array(all_filenames)


# ==============================================================================
# 3. 主程序 (核心修改)
# ==============================================================================
if __name__ == "__main__":
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')
    SELECTED_FILES_CSV = os.path.join(PROCESSED_DIR, 'step1_selected_source_files.csv')

    # 定义全部四个输出文件的路径
    SEGMENTS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'source_segments.npy')
    LABELS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'source_labels.npy')
    RPMS_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'source_rpms.npy')
    FILENAMES_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'source_filenames.npy')  # <--- 【新增】

    if not os.path.exists(SELECTED_FILES_CSV):
        print(f"‼️ 错误：找不到筛选结果文件 {SELECTED_FILES_CSV}。请先运行 01_data_selection.py 脚本。")
    else:
        df_selected = pd.read_csv(SELECTED_FILES_CSV)
        print(f"成功加载筛选文件列表，共 {len(df_selected)} 个文件。")

        # 【修改】接收新增的 filenames 返回值
        segments, labels, rpms, filenames = load_preprocess_and_segment(
            df_selected_files=df_selected,
            target_sr=32000,
            segment_len=3200,
            stride=400
        )

        print(f"\n📊 预处理结果统计:")
        print(f"  - 总样本段数量: {len(segments)}")
        if len(segments) > 0:
            print(f"  - 样本段长度: {segments.shape[1]}")

        # 保存全部四个文件
        np.save(SEGMENTS_OUTPUT_PATH, segments)
        np.save(LABELS_OUTPUT_PATH, labels)
        np.save(RPMS_OUTPUT_PATH, rpms)
        np.save(FILENAMES_OUTPUT_PATH, filenames)  # <--- 【新增】

        print(f"\n💾 预处理后的样本段已保存至: {SEGMENTS_OUTPUT_PATH}")
        print(f"💾 对应的标签已保存至: {LABELS_OUTPUT_PATH}")
        print(f"💾 对应的转速已保存至: {RPMS_OUTPUT_PATH}")
        print(f"💾 对应的文件名已保存至: {FILENAMES_OUTPUT_PATH}")  # <--- 【新增】
        print("\n🎉 02脚本执行完毕，已生成全套预处理文件。")