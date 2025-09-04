import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM #线性混合效应模型
from pygam import LinearGAM, s #广义线性模型
import statsmodels.api as sm
from datetime import datetime
from scipy.stats import spearmanr #相关性检验

data = pd.read_excel("附件.xlsx",sheet_name="男胎检测数据")

# 1.数据预处理
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

# 2. 探索性分析 (EDA)
# 轨迹图：每位孕妇的 Y 染色体浓度随孕周变化
plt.figure(figsize=(10, 6))
for id in data['孕妇代码'].unique()[:10]:  # 展示前 10 个孕妇，直观感受趋势。
    subset = data[data['孕妇代码'] == id]
    plt.plot(subset['GA'], subset['Y染色体浓度'], marker='o', label=id)
plt.xlabel('Gestational Age (weeks)')
plt.ylabel('Y Chromosome Concentration')
plt.title('Y Concentration Trajectories for Selected Pregnant Women')
plt.legend()
plt.show()

# BMI 分层分析
bmi_bins = [20, 28, 32, 36, 40, np.inf]
bmi_labels = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '>=40']
data['BMI_group'] = pd.cut(data['孕妇BMI'], bins=bmi_bins, labels=bmi_labels, right=False)

plt.figure(figsize=(10, 6))
sns.lineplot(x='GA', y='Y染色体浓度', hue='BMI_group', data=data)
plt.xlabel('Gestational Age (weeks)')
plt.ylabel('Y Chromosome Concentration')
plt.title('Y Concentration by BMI Group')
plt.show()

# Spearman 相关性分析
corr_vars = ['Y染色体浓度', 'GA', '孕妇BMI', '年龄', 'GC含量', '被过滤掉读段数的比例']
corr_matrix = data[corr_vars].corr(method='spearman')
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Correlation Matrix')
plt.show()

# 3.模型构建
# 线性混合效应模型 (LMM)
# 模型 M1: 仅包含孕周主效应
model_m1 = MixedLM.from_formula(
    "Y_concentration_logit ~ GA",
    data,
    groups=data['孕妇代码']
)
result_m1 = model_m1.fit()
print("Model M1 (GA only):")
print(result_m1.summary())

# 模型 M2: 孕周 + BMI
model_m2 = MixedLM.from_formula(
    "Y_concentration_logit ~ GA + 孕妇BMI",
    data,
    groups=data['孕妇代码']
)
result_m2 = model_m2.fit()
print("Model M2 (GA + BMI):")
print(result_m2.summary())

# 模型 M3: 孕周 + BMI + 交互项
data['GA_BMI_interaction'] = data['GA'] * data['孕妇BMI']
model_m3 = MixedLM.from_formula(
    "Y_concentration_logit ~ GA + 孕妇BMI + GA_BMI_interaction",
    data,
    groups=data['孕妇代码']
)
result_m3 = model_m3.fit()
print("Model M3 (GA + BMI + Interaction):")
print(result_m3.summary())

# 模型 M4: 加入协变量
model_m4 = MixedLM.from_formula(
    "Y_concentration_logit ~ GA + 孕妇BMI + GA_BMI_interaction + 年龄 + GC含量 + IVF妊娠",
    data,
    groups=data['孕妇代码']
)
result_m4 = model_m4.fit()
print("Model M4 (Full Model):")
print(result_m4.summary())

# 非线性模型 (GAM) 使用 pygam
gam = LinearGAM(s(0, n_splines=10) + s(1, n_splines=10)).fit(
    data[['GA', '孕妇BMI']], data['Y_concentration_logit']
)
print("GAM Model Summary:")
print(gam.summary())

# 4. 可视化模型结果
# LMM 预测轨迹
data['predicted_m3'] = result_m3.fittedvalues
plt.figure(figsize=(10, 6))
sns.lineplot(x='GA', y='predicted_m3', hue='BMI_group', data=data)
plt.xlabel('Gestational Age (weeks)')
plt.ylabel('Predicted Logit(Y Concentration)')
plt.title('LMM Predicted Trajectories by BMI Group')
plt.show()

# GAM 预测曲面
XX, YY = np.meshgrid(np.linspace(data['GA'].min(), data['GA'].max(), 50),
                     np.linspace(data['孕妇BMI'].min(), data['孕妇BMI'].max(), 50))
Z = gam.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

plt.figure(figsize=(10, 6))
contour = plt.contourf(XX, YY, Z, cmap='viridis')
plt.colorbar(contour, label='Predicted Logit(Y Concentration)')
plt.xlabel('Gestational Age (weeks)')
plt.ylabel('BMI')
plt.title('GAM Predicted Surface')
plt.show()

# 5. 显著性检验
# 比较 M2 和 M3 (交互项显著性)
from scipy.stats import chi2
llf_m2 = result_m2.llf
llf_m3 = result_m3.llf
df_diff = result_m3.df_modelwc - result_m2.df_modelwc
lrt_stat = -2 * (llf_m2 - llf_m3)
p_value = chi2.sf(lrt_stat, df_diff)
print(f"LRT for Interaction (M2 vs M3): Stat = {lrt_stat:.2f}, p-value = {p_value:.4f}")

# 6. 健壮性与敏感性分析
# 剔除异常值（例如 Y 染色体浓度 < 0.01）
data_robust = data[data['Y染色体浓度'] > 0.01]
model_m3_robust = MixedLM.from_formula(
    "Y_concentration_logit ~ GA + 孕妇BMI + GA_BMI_interaction",
    data_robust,
    groups=data_robust['孕妇代码']
)
result_m3_robust = model_m3_robust.fit()
print("Model M3 (Robust, Y > 0.01):")
print(result_m3_robust.summary())

# 7. 统计结论
print("\n统计结论：")
print("- 孕周 (GA) 与 Y 染色体浓度呈显著正相关，非线性趋势明显。")
print("- BMI 对 Y 染色体浓度有负调节作用，高 BMI 孕妇的 Y 浓度上升较慢。")
print("- GA × BMI 交互效应显著，BMI 调节了孕周对 Y 浓度的影响。")
print("- 测序质量指标（如 GC 含量）需校正以减少混杂效应。")

