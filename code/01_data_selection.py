import os
import re
import pandas as pd
import numpy as np


# ==============================================================================
# 1. 递归扫描与元数据构建 (完全采纳你验证过的代码)
# ==============================================================================
def build_metadata_from_walk(source_root_dir: str):
    """
    【采纳版】使用os.walk递归扫描所有子文件夹，并从路径和文件名中提取信息。
    """
    records = []
    total_files_found = 0
    unparsed_files = []

    print(f"🔍 开始递归扫描根目录: {os.path.abspath(source_root_dir)}")

    if not os.path.exists(source_root_dir):
        print(f"‼️ 错误：指定的路径不存在！请检查路径: {os.path.abspath(source_root_dir)}")
        return pd.DataFrame()

    for root, dirs, files in os.walk(source_root_dir):
        mat_files_in_this_folder = [f for f in files if f.endswith('.mat')]
        total_files_found += len(mat_files_in_this_folder)

        for filename in mat_files_in_this_folder:
            file_path = os.path.join(root, filename)
            path_lower = file_path.lower().replace('\\', '/')

            sampling_rate = '48k' if '48khz' in path_lower else '12k'
            sensor = 'DE' if 'de_data' in path_lower or 'normal_data' in path_lower else \
                'FE' if 'fe_data' in path_lower else \
                    'BA' if 'ba_data' in path_lower else 'unknown'

            fault_type, fault_size, load = None, None, None
            if 'normal_data' in path_lower:
                fault_type, fault_size, load = 'N', '000', '0'
            else:
                match = re.match(r'([A-Z]+)(\d{3})(?:@\d+)?_(\d).*?\.mat', filename, re.IGNORECASE)
                if match:
                    ft, fs, ld = match.groups()
                    fault_type, fault_size, load = ft.upper(), fs, ld

            if fault_type:
                records.append({
                    'filename': filename, 'file_path': file_path,
                    'sampling_rate': sampling_rate, 'sensor': sensor,
                    'fault_type': fault_type, 'fault_size': fault_size, 'load': load
                })
            else:
                unparsed_files.append(file_path)

    print(f"✅ 扫描完成，总共找到 {total_files_found} 个 .mat 文件。")
    if unparsed_files:
        print(f"⚠️  其中有 {len(unparsed_files)} 个文件因命名不规范无法解析。")

    return pd.DataFrame(records)


# ==============================================================================
# 2. 【核心修改】最终版数据筛选策略函数
# ==============================================================================
def apply_final_selection_strategy(df_all_files: pd.DataFrame):
    """
    执行我们最终确定的筛选策略：
    1. 正常样本：全部选取。
    2. 故障样本：只保留驱动端(DE)、载荷为1或2马力的样本。
    """
    print("\n🚀 开始执行最终版文件筛选策略...")

    # 策略1：选取所有正常(N)样本
    df_normal = df_all_files[df_all_files['fault_type'] == 'N'].copy()
    print(f"  - [正常样本] 已选择全部 {len(df_normal)} 个正常文件。")

    # 策略2：筛选故障样本
    # 先选出所有故障文件
    df_faults_initial = df_all_files[df_all_files['fault_type'] != 'N'].copy()

    # 筛选条件 a: 传感器位置必须是 'DE'
    df_faults_de = df_faults_initial[df_faults_initial['sensor'] == 'DE']
    print(
        f"  - [故障样本-传感器] 从 {len(df_faults_initial)} 个故障文件中, 筛选出 {len(df_faults_de)} 个驱动端(DE)文件。")

    # 筛选条件 b: 载荷必须是 '1' 或 '2'
    df_faults_final = df_faults_de[df_faults_de['load'].isin(['1', '2'])]
    print(f"  - [故障样本-载荷] 进一步筛选出 {len(df_faults_final)} 个载荷为1或2马力的文件。")

    # 合并最终选择的正常样本和故障样本
    df_selected = pd.concat([df_normal, df_faults_final], ignore_index=True)

    print("\n✅ 最终筛选策略执行完毕！")
    return df_selected


# ==============================================================================
# 3. 主程序
# ==============================================================================
if __name__ == "__main__":
    # 定义源数据根目录
    SOURCE_ROOT_DIR = os.path.join('..', 'data', 'source')

    # 1. 使用你验证过的函数，扫描并解析所有文件
    df_all_files = build_metadata_from_walk(SOURCE_ROOT_DIR)

    if not df_all_files.empty:
        print(f"\n📊 成功解析 {len(df_all_files)} 个文件。")

        # 2. 应用我们最终确定的筛选策略
        df_selected = apply_final_selection_strategy(df_all_files)

        print("\n📊 最终选定的文件概览 (按故障类型统计):")
        print(df_selected['fault_type'].value_counts())

        print("\n(按载荷统计):")
        print(df_selected['load'].value_counts())

        # 3. 保存结果
        output_dir = os.path.join('..', 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'step1_selected_source_files.csv')
        df_selected.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"\n💾 最终筛选结果已保存至: {os.path.abspath(output_path)}")
        print("\n🎉 任务一数据筛选工作已按最终方案完成！")
    else:
        print("\n未找到任何数据文件，程序已终止。")