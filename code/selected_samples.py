# 文件: 2025HWB/code/selected_samples.py
"""
本文件定义了用于后续建模的样本列表。
所有队员在进行特征提取、模型训练等操作时，都应使用此列表中的文件名。
"""

import pandas as pd
import os


def load_data_overview():
    """加载数据概览表"""
    csv_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "source_data_overview.csv"))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据概览表不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"📊 成功加载数据概览表，共 {len(df)} 个样本。")
    return df


def select_representative_samples(df):
    """根据筛选原则，从数据概览表中选出代表性样本"""
    selected = []
    print("\n🔍 开始筛选代表性样本...")

    # --- 1. Normal (N) 样本：全选 ---
    normal_df = df[df['故障类型'] == 'N'].copy()
    if len(normal_df) == 0:
        raise ValueError("未找到任何正常样本 (N)！")
    selected.extend(normal_df['文件名'].tolist())
    print(f"✅ Normal (N): 全选 {len(normal_df)} 个样本。")

    # --- 2. Ball Fault (B) 样本：选15个 ---
    def select_samples_for_fault_type(fault_df, fault_name, target_count=15):
        """为指定故障类型选择样本的通用函数"""
        selected_list = []

        # 第一阶段：筛选 12k_DE 数据 (硬性要求)
        stage1_df = fault_df[fault_df['文件名'].str.contains('12k_DE')].copy()
        if len(stage1_df) == 0:
            print(f"⚠️  警告: 未找到任何 12k_DE 数据！")
            stage1_df = fault_df.copy()  # 降级处理，使用所有数据

        # 第二阶段：优先选择 1730-1750 RPM，如果不够，使用所有转速
        stage2_df = stage1_df[(stage1_df['转速(RPM)'] >= 1730) & (stage1_df['转速(RPM)'] <= 1750)].copy()
        if len(stage2_df) < target_count * 0.5:  # 如果优质样本太少，放宽条件
            print(f"  ⚠️  {fault_name}: 1730-1750 RPM 样本不足，放宽至所有转速。")
            stage2_df = stage1_df.copy()

        # 第三阶段：按故障尺寸和载荷分层抽样
        for size in ['007', '014', '021', '028']:
            if len(selected_list) >= target_count:
                break
            size_df = stage2_df[stage2_df['故障尺寸'] == size]
            for load in ['0', '1', '2', '3']:
                if len(selected_list) >= target_count:
                    break
                load_df = size_df[size_df['载荷'] == load]
                if len(load_df) > 0:
                    # 选第一个符合条件的样本
                    selected_list.append(load_df.iloc[0]['文件名'])
                    print(f"  ✅ 选中 {fault_name}: 尺寸{size}, 载荷{load} -> {load_df.iloc[0]['文件名']}")

        # 第四阶段：如果还不够，随机补充
        if len(selected_list) < target_count:
            remaining_df = stage2_df[~stage2_df['文件名'].isin(selected_list)]
            # 按转速排序，优先选接近1740 RPM的
            remaining_df = remaining_df.sort_values('转速(RPM)', key=lambda x: abs(x - 1740))
            for _, row in remaining_df.iterrows():
                if len(selected_list) >= target_count:
                    break
                selected_list.append(row['文件名'])
                print(f"  ➕ 补充选中 {fault_name}: {row['文件名']} (转速: {row['转速(RPM)']} RPM)")

        return selected_list[:target_count]

    # 为 Ball Fault 选择样本
    ball_df = df[df['故障类型'] == 'B'].copy()
    selected_ball = select_samples_for_fault_type(ball_df, "Ball Fault", 15)
    selected.extend(selected_ball)
    print(f"✅ Ball Fault (B): 已选 {len(selected_ball)} 个样本。")

    # 为 Inner Race Fault 选择样本
    inner_df = df[df['故障类型'] == 'I'].copy()
    selected_inner = select_samples_for_fault_type(inner_df, "Inner Race", 15)
    selected.extend(selected_inner)
    print(f"✅ Inner Race Fault (IR): 已选 {len(selected_inner)} 个样本。")

    # 为 Outer Race Fault 选择样本
    outer_df = df[df['故障类型'] == 'O'].copy()
    selected_outer = select_samples_for_fault_type(outer_df, "Outer Race", 15)
    selected.extend(selected_outer)
    print(f"✅ Outer Race Fault (OR): 已选 {len(selected_outer)} 个样本。")

    print(f"\n🎉 样本筛选完成！共选出 {len(selected)} 个样本。")
    return selected


# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 加载数据
    df_overview = load_data_overview()

    # 执行筛选
    SELECTED_FILES = select_representative_samples(df_overview)

    # 打印最终列表
    print("\n📋 最终选定的样本列表:")
    for i, fname in enumerate(SELECTED_FILES, 1):
        print(f"  {i:2d}. {fname}")