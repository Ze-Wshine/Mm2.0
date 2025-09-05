import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM #线性混合效应模型
from pygam import LinearGAM, s #广义线性模型
import statsmodels.api as sm
from datetime import datetime
from scipy.stats import spearmanr #相关性检验

plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体，支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块

data = pd.read_excel("附件.xlsx",sheet_name="男胎检测数据")

# 数据预处理
# 转换孕周函数（小数表示）
def convert_gestational_age(ga_str):
    if isinstance(ga_str, str):  # 检查是否为字符串
        try:
            # 将大写 W 转换为小写 w，确保兼容大小写
            ga_str = ga_str.lower()
            # 分割字符串，提取周数和天数
            weeks_part = ga_str.split('w')[0].strip()
            weeks = int(weeks_part)  # 转换为整数
            # 检查是否有天数部分
            days = 0
            if '+' in ga_str:
                days_part = ga_str.split('+')[1].strip()
                days = int(days_part) if days_part else 0
            # 确保周数和天数合理
            if weeks >= 0 and 0 <= days < 7:
                return weeks + days / 7
            else:
                return np.nan
        except (ValueError, IndexError):
            return np.nan
    return np.nan

# 读取并清洗列名
data = pd.read_excel("附件.xlsx", sheet_name="男胎检测数据")
# 去掉首尾空白与换行
data.columns = data.columns.str.strip().str.replace("\n", "", regex=True)
print("Columns in sheet:", data.columns.tolist())
# 计算孕周数值
data['GA'] = data['检测孕周'].apply(convert_gestational_age)

# 对 Y 染色体浓度进行 logit 变换
epsilon = 1e-6  # 避免溢出
data['Y_concentration_logit'] = np.log(data['Y染色体浓度'] / (1 - data['Y染色体浓度'] + epsilon))

# 处理日期（末次月经时间和检测日期，后续计算时间差）
data['末次月经'] = pd.to_datetime(data['末次月经'])
data['检测日期'] = pd.to_datetime(data['检测日期'])

# 检查缺失值
print("缺失值检查：")
print(data.isnull().sum())