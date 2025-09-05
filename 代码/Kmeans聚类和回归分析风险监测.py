import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

data = pd.read_excel("附件.xlsx", sheet_name="男胎检测数据")

def convert_gestational_age(ga_str):
    if isinstance(ga_str, str):
        try:
            ga_str = ga_str.lower().replace(" ", "")
            weeks = int(ga_str.split('w')[0])
            days = int(ga_str.split('+')[1]) if '+' in ga_str else 0
            if weeks >= 0 and 0 <= days < 7:
                return weeks + days / 7
            return np.nan
        except (ValueError, IndexError):
            return np.nan
    return np.nan

data['GA'] = data['检测孕周'].apply(convert_gestational_age)
data['Y_reached'] = (data['Y染色体浓度'] >= 0.04).astype(int)

# 过滤异常值
data = data[(data['GC含量'] >= 0.4) & (data['GC含量'] <= 0.6)]


first_reached = data[data['Y_reached'] == 1].groupby('孕妇代码')['GA'].min().reset_index()
first_reached = first_reached.rename(columns={'GA': 'GA_first'})
data_unique = data.drop_duplicates('孕妇代码')[['孕妇代码', '孕妇BMI']].merge(first_reached, on='孕妇代码')


X_cluster = data_unique[['孕妇BMI', 'GA_first']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)


inertias = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
plt.plot(range(2, 7), inertias, marker='o')
plt.xlabel('簇数 k')
plt.ylabel('组内平方和')
plt.title('K-means 聚类肘部法则图')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
data_unique['Cluster'] = kmeans.fit_predict(X_scaled)

bmi_edges = pd.qcut(data_unique['孕妇BMI'], q=4)
data_unique['BMI_group'] = bmi_edges
bmi_ranges = data_unique.groupby('BMI_group')['孕妇BMI'].agg(BMI_Min='min', BMI_Max='max')

X = data[['GA', '孕妇BMI']].dropna()
y = data['Y_reached'].dropna()
model_lr = LogisticRegression().fit(X, y)

def risk_function_updated(ga, bmi, model, w1=0.7, w2=0.3):
    X_pred = pd.DataFrame({'GA': [ga], '孕妇BMI': [bmi]})
    prob_false_negative = 1 - model.predict_proba(X_pred)[:, 1]
    delay_penalty = ((ga - 12) / 15) ** 2
    return w1 * prob_false_negative + w2 * delay_penalty

ga_grid = np.arange(10, 25, 0.1)

quantiles_80 = data_unique.groupby('BMI_group')['GA_first'].quantile(0.8).rename('NIPT_80_Quantile')
risk_results_updated = {}
for group in data_unique['BMI_group'].unique():
    bmi_mean = data_unique[data_unique['BMI_group'] == group]['孕妇BMI'].mean()
    risks = [risk_function_updated(ga, bmi_mean, model_lr) for ga in ga_grid]
    optimal_ga = ga_grid[np.argmin(risks)]
    risk_results_updated[group] = optimal_ga

results_table = pd.DataFrame({
    'BMI_Min': bmi_ranges['BMI_Min'],
    'BMI_Max': bmi_ranges['BMI_Max'],
    'NIPT_80_Quantile': quantiles_80,
    'Risk_Optimal_GA': pd.Series(risk_results_updated)
}).sort_index()

print("\n连续 BMI 区间分组及 NIPT 时点：\n")
print(results_table)

data['GC_deviation'] = np.abs(data['GC含量'] - 0.5)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GC_deviation', y='Y染色体浓度', hue='Y_reached', data=data)
plt.xlabel('GC 含量偏差')
plt.ylabel('Y 染色体浓度')
plt.title('GC 含量偏差对 Y 染色体浓度的影响')
plt.show()
