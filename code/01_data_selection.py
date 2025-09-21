import os
import re
import pandas as pd
import numpy as np
#数据分析与筛选

# ==============================================================================
# 1. 递归扫描与元数据构建 (再次修正正则表达式)
# ==============================================================================
def build_metadata_from_walk(source_root_dir: str):
    """
    【最终修正版】更新正则表达式，使其能够兼容所有已知的文件命名格式。
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
                # ==================== 最终核心修改在此行 ====================
                # 新的正则表达式可以处理 OR007@6_0.mat, B007_0.mat, B028_0_(1797rpm).mat 等所有格式
                match = re.match(r'([A-Z]+)(\d{3})(?:@\d+)?_(\d).*?\.mat', filename, re.IGNORECASE)
                # ===========================================================

                if match:
                    # 注意：因为增加了一个非捕获组 (?:@\d+)?, 分组索引不变
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
        print(f"⚠️  其中有 {len(unparsed_files)} 个文件因命名不规范无法解析，列表如下:")
        for f in unparsed_files:
            print(f"  - {f}")

    return pd.DataFrame(records)


# ==============================================================================
# 2. 保证数量的数据筛选策略 (保持不变)
# ==============================================================================
def select_files_guaranteed_count(df: pd.DataFrame):
    """
    【无需修改】根据“多样性优先，数量保证”原则筛选文件。
    """
    print("\n🚀 开始执行“多样性优先，数量保证”文件筛选策略...")

    df_fault_pool = df[(df['sampling_rate'] == '12k') & (df['sensor'] == 'DE')].copy()
    df_n = df[df['fault_type'] == 'N'].copy()
    print(f"  - [正常 N]: 目标选取4个，实际选取 {len(df_n)} 个。")

    selection_targets = {
        'IR': [('007', '1'), ('014', '2'), ('021', '3'), ('028', '0'), ('007', '0')],
        'B': [('007', '1'), ('014', '2'), ('021', '3'), ('028', '0'), ('007', '0')],
        'OR': [('007', '1'), ('014', '2'), ('021', '3'), ('007', '0'), ('014', '0')]
    }

    final_selected_dfs = [df_n]
    for f_type, targets in selection_targets.items():
        target_count = 12
        df_pool = df_fault_pool[df_fault_pool['fault_type'] == f_type]

        primary_selection_list = []
        for size, load in targets:
            match = df_pool[(df_pool['fault_size'] == size) & (df_pool['load'] == load)]
            if not match.empty:
                primary_selection_list.append(match.iloc[[0]])

        df_primary = pd.concat(primary_selection_list, ignore_index=True) if primary_selection_list else pd.DataFrame()

        current_count = len(df_primary)
        if current_count < target_count:
            print(f"  - [{f_type}]: 多样性初选得到 {current_count} 个，需要补充 {target_count - current_count} 个。")
            remaining_indices = df_pool.index[~df_pool.index.isin(df_primary.index)]
            df_remaining = df_pool.loc[remaining_indices]

            num_to_add = target_count - current_count
            df_supplement = df_remaining.head(num_to_add)
            df_final_fault = pd.concat([df_primary, df_supplement], ignore_index=True)
        else:
            df_final_fault = df_primary.head(target_count)

        final_selected_dfs.append(df_final_fault)
        print(f"  - [{f_type}]: 最终选取 {len(df_final_fault)} 个文件。")

    final_df = pd.concat(final_selected_dfs, ignore_index=True)
    print("\n✅ 筛选策略执行完毕！")
    return final_df


# ==============================================================================
# 3. 主程序 (保持不变)
# ==============================================================================
if __name__ == "__main__":
    SOURCE_ROOT_DIR = os.path.join('..', 'data', 'source')

    # 1. 【调用最终版函数】创建元数据表
    df_all_files = build_metadata_from_walk(SOURCE_ROOT_DIR)

    if not df_all_files.empty:
        print(f"\n📊 成功解析 {len(df_all_files)} 个文件。")

        # 2. 执行筛选
        df_selected = select_files_guaranteed_count(df_all_files)

        print("\n📊 最终选定的文件概览 (按故障类型统计):")
        print(df_selected['fault_type'].value_counts())

        # 3. 保存结果
        output_dir = os.path.join('..', 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'step1_selected_source_files.csv')
        df_selected.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"\n💾 筛选结果已保存至: {output_path}")
        print("\n🎉 步骤一：数据分析与筛选完成！")
    else:
        print("\n未找到任何数据文件，程序已终止。")