import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM 
from pygam import LinearGAM, s
import statsmodels.api as sm
from datetime import datetime
from scipy.stats import spearmanr 

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    

data = pd.read_excel("附件.xlsx",sheet_name="男胎检测数据")


def convert_gestational_age(ga_str):
    if isinstance(ga_str, str):  
        try:
            ga_str = ga_str.lower()
            weeks_part = ga_str.split('w')[0].strip()
            weeks = int(weeks_part)  
            days = 0
            if '+' in ga_str:
                days_part = ga_str.split('+')[1].strip()
                days = int(days_part) if days_part else 0
            if weeks >= 0 and 0 <= days < 7:
                return weeks + days / 7
            else:
                return np.nan
        except (ValueError, IndexError):
            return np.nan
    return np.nan


data = pd.read_excel("附件.xlsx", sheet_name="男胎检测数据")

data.columns = data.columns.str.strip().str.replace("\n", "", regex=True)
print("Columns in sheet:", data.columns.tolist())

data['GA'] = data['检测孕周'].apply(convert_gestational_age)


epsilon = 1e-6  
data['Y_concentration_logit'] = np.log(data['Y染色体浓度'] / (1 - data['Y染色体浓度'] + epsilon))


data['末次月经'] = pd.to_datetime(data['末次月经'])
data['检测日期'] = pd.to_datetime(data['检测日期'])


print("缺失值检查：")
print(data.isnull().sum())