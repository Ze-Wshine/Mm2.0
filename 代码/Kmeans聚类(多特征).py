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

data = data[(data['GC含量'] >= 0.4) & (data['GC含量'] <= 0.6)]

first_reached = data[data['Y_reached'] == 1].groupby('孕妇代码')['GA'].min().reset_index()
first_reached = first_reached.rename(columns={'GA': 'GA_first'})

data_unique = data.drop_duplicates('孕妇代码')[['孕妇代码', '孕妇BMI', '身高', '体重', '年龄']].merge(first_reached, on='孕妇代码', how='left')

features = ['孕妇BMI', 'GA_first', '身高', '体重', '年龄']
X_cluster = data_unique[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42)
data_unique.loc[X_cluster.index, 'Cluster'] = kmeans.fit_predict(X_scaled)

cluster_order = data_unique.groupby('Cluster')['孕妇BMI'].mean().sort_values().index
cluster_map = {old: new for new, old in enumerate(cluster_order)}
data_unique['Cluster_ordered'] = data_unique['Cluster'].map(cluster_map)

cluster_stats_ordered = data_unique.groupby('Cluster_ordered').agg({
    '孕妇BMI': ['min', 'max', 'mean'],
    '身高': 'mean',
    '体重': 'mean',
    '年龄': 'mean',
}).reset_index()
cluster_stats_ordered.columns = ['Cluster', 'BMI_Min', 'BMI_Max', 'BMI_Mean', 'Height_Mean', 'Weight_Mean', 'Age_Mean']

df_model = data[['GA', '孕妇BMI', 'Y_reached']].dropna()
X = df_model[['GA', '孕妇BMI']]
y = df_model['Y_reached']
model_lr = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)

def risk_function_updated(ga, bmi, model, w1=0.7, w2=0.3):
    if np.isnan(ga) or np.isnan(bmi):
        return np.inf
    X_pred = pd.DataFrame({'GA': [ga], '孕妇BMI': [bmi]})
    prob_false_negative = 1 - model.predict_proba(X_pred)[:, 1]
    delay_penalty = ((ga - 12) / 15) ** 2
    return w1 * prob_false_negative + w2 * delay_penalty

ga_grid = np.arange(10, 25, 0.1)

risk_results_updated = {}
for cluster in data_unique['Cluster_ordered'].unique():
    bmi_mean = data_unique[data_unique['Cluster_ordered'] == cluster]['孕妇BMI'].mean()
    if np.isnan(bmi_mean):
        continue
    risks = [risk_function_updated(ga, bmi_mean, model_lr) for ga in ga_grid]
    optimal_ga = ga_grid[np.argmin(risks)]
    risk_results_updated[cluster] = optimal_ga

quantiles_80 = data_unique.groupby('Cluster_ordered')['GA_first'].quantile(0.8).rename('NIPT_80_Quantile')

results_table_ordered = pd.DataFrame({
    'Cluster': quantiles_80.index,
    'BMI_Min': cluster_stats_ordered['BMI_Min'],
    'BMI_Max': cluster_stats_ordered['BMI_Max'],
    'NIPT_80_Quantile': quantiles_80.values,
    'Risk_Optimal_GA': pd.Series(risk_results_updated)
}).sort_values('Cluster').reset_index(drop=True)

print("\n第三问聚类分组及 NIPT 时点（按 BMI 连续排序）：\n")
print(results_table_ordered)

data['GC_deviation'] = np.abs(data['GC含量'] - 0.5)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GC_deviation', y='Y染色体浓度', hue='Y_reached', data=data)
plt.xlabel('GC 含量偏差')
plt.ylabel('Y 染色体浓度')
plt.title('GC 含量偏差对 Y 染色体浓度的影响')
plt.show()
