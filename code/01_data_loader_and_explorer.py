
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List, Tuple

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_mat_file(file_path: str) -> Dict[str, Any]:
    """
    读取 .mat 文件，自动处理 v7.3 和旧版本格式。
    """
    try:
        # 尝试用 scipy 读取（适用于 v7.3 之前的文件）
        mat_data = scipy.io.loadmat(file_path)
        print(f"文件 {os.path.basename(file_path)} 已用 scipy.io.loadmat 成功读取。")
        return mat_data
    except NotImplementedError:
        # 如果报 NotImplementedError，说明是 v7.3+ 格式，用 h5py 读取
        print(f"文件 {os.path.basename(file_path)} 是 v7.3+ 格式，正在使用 h5py 读取...")
        import h5py
        mat_data = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                data = np.array(f[key])
                if data.ndim == 2 and data.shape[0] == 1:
                    data = data.flatten()
                mat_data[key] = data
        return mat_data
    except Exception as e:
        print(f"读取文件 {os.path.basename(file_path)} 时发生错误: {e}")
        return None


def load_all_data(folder_path: str) -> Dict[str, Dict[str, Any]]:
    """
    递归读取指定文件夹及其所有子文件夹下的所有 .mat 文件。
    """
    all_data = {}
    total_files_found = 0

    print(f"🔍 开始递归扫描文件夹: {os.path.abspath(folder_path)}")

    # 使用 os.walk 递归遍历
    for root, dirs, files in os.walk(folder_path):
        print(f"  📁 正在扫描: {root}")
        mat_files_in_this_folder = [f for f in files if f.endswith('.mat')]
        total_files_found += len(mat_files_in_this_folder)

        if mat_files_in_this_folder:
            print(f"    📥 发现 {len(mat_files_in_this_folder)} 个 .mat 文件")

        for file_name in mat_files_in_this_folder:
            file_path = os.path.join(root, file_name)
            # --- 修复版：生成全局唯一键名 ---
            # 1. 提取采样率
            if '12kHz' in root:
                sample_rate = '12k'
            elif '48kHz' in root:
                sample_rate = '48k'
            else:
                sample_rate = 'unknown'

            # 2. 提取传感器位置 (DE, FE, BA)
            # --- 修复版：提取传感器位置 ---
            sensor_pos = 'unknown'

            # 优先判断是否为 Normal_data
            if 'Normal_data' in root:
                # 正常样本通常来自驱动端，所以标记为 'DE'
                sensor_pos = 'DE'
            elif 'DE_data' in root:
                sensor_pos = 'DE'
            elif 'FE_data' in root:
                sensor_pos = 'FE'
            elif 'BA_data' in root:
                sensor_pos = 'BA'

            # 3. 组合唯一键名：文件名_采样率_传感器位置
            base_name = f"{os.path.splitext(file_name)[0]}_{sample_rate}_{sensor_pos}"

            # 避免重复加载同名文件（虽然概率很低）
            if base_name in all_data:
                print(f"    ⚠️  警告: 文件名冲突！'{base_name}' 已存在，跳过 {file_path}")
                continue

            data = load_mat_file(file_path)
            if data is not None:
                all_data[base_name] = data
                print(f"    ✅ 成功加载: {base_name}")

    print(f"\n🎉 总共扫描到 {total_files_found} 个 .mat 文件，成功加载 {len(all_data)} 个。")
    return all_data


def explore_data_structure(data_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    探索数据结构，生成一个数据概览表。
    """
    records = []
    for file_key, data in data_dict.items():
        # 提取文件名信息 (如 B007_0)
        fault_type = file_key[0]  # 'N', 'O', 'I', 'B'
        fault_size = file_key[1:4] if len(file_key) > 1 else 'N/A'  # '007', '014' 等
        load_condition = file_key[-1] if file_key[-1].isdigit() else 'N/A'  # '0', '1', '2', '3'

        # 检查包含哪些传感器数据
        has_de = any('DE_time' in key for key in data.keys())
        has_fe = any('FE_time' in key for key in data.keys())
        has_ba = any('BA_time' in key for key in data.keys())

        # 获取 RPM
        rpm_keys = [key for key in data.keys() if 'RPM' in key]
        rpm = data[rpm_keys[0]].item() if rpm_keys and data[rpm_keys[0]].size > 0 else np.nan

        # 获取 DE_time 长度 (如果存在)
        de_keys = [key for key in data.keys() if 'DE_time' in key]
        de_length = len(data[de_keys[0]]) if de_keys else 0

        records.append({
            '文件名': file_key,
            '故障类型': fault_type,
            '故障尺寸': fault_size,
            '载荷': load_condition,
            '包含DE': has_de,
            '包含FE': has_fe,
            '包含BA': has_ba,
            '转速(RPM)': rpm,
            'DE数据长度': de_length
        })

    df = pd.DataFrame(records)
    return df


def visualize_sample(data_dict: Dict[str, Dict[str, Any]], file_key: str, save_path: str = None):
    """
    可视化单个样本的时域和频域图。
    """
    data = data_dict[file_key]

    # 找到 DE_time 数据
    de_keys = [key for key in data.keys() if 'DE_time' in key]
    if not de_keys:
        print(f"文件 {file_key} 中未找到 DE_time 数据。")
        return
    de_time_series = data[de_keys[0]].flatten()

    # 获取 RPM
    rpm_keys = [key for key in data.keys() if 'RPM' in key]
    rpm = data[rpm_keys[0]].item() if rpm_keys else 0

    # 计算采样频率 (假设为 12KHz，根据附件1)
    fs = 12000  # Hz
    n = len(de_time_series)
    t = np.arange(n) / fs  # 时间轴

    # 计算 FFT
    fft_vals = np.fft.fft(de_time_series)
    fft_freq = np.fft.fftfreq(n, 1 / fs)
    # 只取正频率部分
    positive_freq_indices = np.where(fft_freq >= 0)
    fft_freq_pos = fft_freq[positive_freq_indices]
    fft_vals_pos = np.abs(fft_vals[positive_freq_indices])

    # 创建子图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'样本 {file_key} 分析 (RPM: {rpm})', fontsize=16)

    # 时域图
    axes[0].plot(t[:1000], de_time_series[:1000])  # 只画前1000点，避免太密集
    axes[0].set_title('时域波形 (前1000点)')
    axes[0].set_xlabel('时间 (秒)')
    axes[0].set_ylabel('加速度')
    axes[0].grid(True)

    # 频域图
    axes[1].plot(fft_freq_pos[:2000], fft_vals_pos[:2000])  # 只画前2000个频率点
    axes[1].set_title('频谱图 (幅值)')
    axes[1].set_xlabel('频率 (Hz)')
    axes[1].set_ylabel('幅值')
    axes[1].grid(True)

    # 根据故障类型，计算并标出理论故障频率
    fault_type = file_key[0]
    if fault_type in ['O', 'I', 'B'] and rpm > 0:
        # 轴承参数 (根据附件1, SKF6205 for DE)
        n_balls = 9
        ball_diameter = 0.3126  # inches
        pitch_diameter = 1.537  # inches
        contact_angle = 0  # degrees, assumed 0 for simplicity

        # 计算特征频率
        f_r = rpm / 60  # 旋转频率 (Hz)
        bpfo = (n_balls / 2) * f_r * (1 - (ball_diameter / pitch_diameter) * np.cos(np.deg2rad(contact_angle)))
        bpfi = (n_balls / 2) * f_r * (1 + (ball_diameter / pitch_diameter) * np.cos(np.deg2rad(contact_angle)))
        bsf = (pitch_diameter / (2 * ball_diameter)) * f_r * (
                    1 - ((ball_diameter / pitch_diameter) * np.cos(np.deg2rad(contact_angle))) ** 2)

        if fault_type == 'O':
            axes[1].axvline(x=bpfo, color='r', linestyle='--', label=f'BPFO={bpfo:.1f}Hz')
        elif fault_type == 'I':
            axes[1].axvline(x=bpfi, color='g', linestyle='--', label=f'BPFI={bpfi:.1f}Hz')
        elif fault_type == 'B':
            axes[1].axvline(x=bsf, color='b', linestyle='--', label=f'BSF={bsf:.1f}Hz')
        axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 图表已保存至: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# --- 主执行部分 ---
if __name__ == "__main__":
    # 1. 加载源域数据
    source_data_path = "../data/source/"
    source_data_dict = load_all_data(source_data_path)

    # 2. 生成数据概览表
    df_overview = explore_data_structure(source_data_dict)
    print("\n📊 数据概览表:")
    print(df_overview.head(10))  # 打印前10行

    # 保存数据概览表到 processed 文件夹
    os.makedirs("../data/processed/", exist_ok=True)
    df_overview.to_csv("../data/processed/source_data_overview.csv", index=False, encoding='utf-8-sig')
    print("💾 数据概览表已保存至: ../data/processed/source_data_overview.csv")

    # 3. 可视化典型样本
    # 选择每类故障的一个代表性样本
    sample_files = [
        'N_0_48k_DE',  # Normal
        'B007_0_12k_DE',  # Ball Fault
        'IR007_0_12k_DE',  # Inner Race Fault
        'OR007@6_0_12k_DE'  # Outer Race Fault
    ]

    for file_key in sample_files:
        if file_key in source_data_dict:
            save_path = f"../data/processed/问题1-图-{file_key}.png"
            visualize_sample(source_data_dict, file_key, save_path)
        else:
            print(f"⚠️  未找到文件: {file_key}")

    print("\n🎉 数据预处理第一步完成！请检查 data/processed/ 文件夹中的输出。")