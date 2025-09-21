# 文件: 2025HWB/code/config.py
"""
项目配置文件，包含所有关键参数和数据集定义。
"""

# ========== 核心数据集配置 ==========
# 用于源域建模的49个精选样本
SOURCE_DOMAIN_SELECTED_SAMPLES = [
    'N_0_48k_DE',
    'N_1_(1772rpm)_48k_DE',
    'N_2_(1750rpm)_48k_DE',
    'N_3_48k_DE',
    'B007_2_12k_DE',
    'B014_2_12k_DE',
    'B021_3_12k_DE',
    'B021_2_12k_DE',
    'B014_3_12k_DE',
    'B007_3_12k_DE',
    'B007_1_12k_DE',
    'B014_1_12k_DE',
    'B021_1_12k_DE',
    'B007_0_12k_DE',
    'B014_0_12k_DE',
    'B021_0_12k_DE',
    'B028_0_(1797rpm)_12k_DE',
    'B028_1_(1772rpm)_12k_DE',
    'B028_2_(1750rpm)_12k_DE',
    'IR007_2_12k_DE',
    'IR014_2_12k_DE',
    'IR014_3_12k_DE',
    'IR021_2_12k_DE',
    'IR021_3_12k_DE',
    'IR007_3_12k_DE',
    'IR007_1_12k_DE',
    'IR014_1_12k_DE',
    'IR021_1_12k_DE',
    'IR014_0_12k_DE',
    'IR007_0_12k_DE',
    'IR021_0_12k_DE',
    'IR028_0_(1797rpm)_12k_DE',
    'IR028_1_(1772rpm)_12k_DE',
    'IR028_2_(1750rpm)_12k_DE',
    'OR021@12_2_12k_DE',
    'OR021@3_2_12k_DE',
    'OR021@6_2_12k_DE',
    'OR014@6_2_12k_DE',
    'OR007@6_2_12k_DE',
    'OR007@12_2_12k_DE',
    'OR007@3_2_12k_DE',
    'OR007@6_3_12k_DE',
    'OR007@3_3_12k_DE',
    'OR007@12_3_12k_DE',
    'OR014@6_3_12k_DE',
    'OR021@6_3_12k_DE',
    'OR021@3_3_12k_DE',
    'OR021@12_3_12k_DE',
    'OR021@6_1_12k_DE',
]

# 目标域文件列表 (16个未知标签文件)
TARGET_DOMAIN_FILES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'
]

# ========== 特征提取参数 ==========
SAMPLE_RATE = 12000  # 采样率 12kHz
N_FFT = 1024         # FFT点数
N_MFCC = 13          # MFCC系数个数 (如果用)

# ========== 模型参数 ==========
RANDOM_SEED = 42
TEST_SIZE = 0.2

# ========== 路径配置 ==========
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SOURCE_DATA_DIR = os.path.join(DATA_DIR, 'source')
TARGET_DATA_DIR = os.path.join(DATA_DIR, 'target')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 创建结果文件夹
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)