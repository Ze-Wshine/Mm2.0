import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_excel("附件.xlsx", sheet_name="女胎检测数据")

def parse_ga(ga_str):
    if not isinstance(ga_str, str):
        return np.nan
    ga_str = ga_str.lower().replace(" ", "")
    if "w" not in ga_str:
        return np.nan
    try:
        weeks_part = ga_str.split("w")[0]
        weeks = float(weeks_part)
        days = 0
        if "+" in ga_str:
            days_part = ga_str.split("+")[1]
            days = float(days_part)
        return weeks + days / 7
    except (ValueError, IndexError):
        return np.nan

data['GA'] = data['检测孕周'].apply(parse_ga)

data['GC偏差'] = np.abs(data['GC含量'] - 0.5)
data['X染色体Z值绝对值'] = np.abs(data['X染色体的Z值'])
data['唯一比对读段数_log'] = np.log1p(data['唯一比对的读段数'])

data['检测成功'] = (data['X染色体Z值绝对值'] < 3).astype(int)

data = data[(data['GC含量'] >= 0.4) & (data['GC含量'] <= 0.6) & 
            (data['GA'] >= 10) & (data['GA'] <= 25)].copy()

data_unique = data.sort_values(['孕妇代码', 'GA']).groupby('孕妇代码').first().reset_index()

data_unique['异常'] = data_unique[['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值']].apply(
    lambda row: int(any(row.abs() > 3)), axis=1
)

features = ['孕妇BMI', '年龄', '身高', '体重']
X_cluster = data_unique[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42)
data_unique.loc[X_cluster.index, 'BMI分组'] = kmeans.fit_predict(X_scaled)

bmi_ranges = data_unique.groupby('BMI分组')['孕妇BMI'].agg(['min', 'max', 'mean']).reset_index()
print("BMI 分组范围：\n", bmi_ranges)

plt.figure(figsize=(10, 6))
for cluster in sorted(data_unique['BMI分组'].dropna().unique()):
    mask = data_unique['BMI分组'] == cluster
    cluster_data = data_unique[mask].sort_values('GA')
    ga_sorted = cluster_data['GA']
    event_sorted = cluster_data['检测成功']
    survival_prob = 1 - np.cumsum(event_sorted) / len(event_sorted)
    plt.step(ga_sorted, survival_prob, where='post', label=f'BMI {bmi_ranges.loc[cluster,"min"]:.1f}-{bmi_ranges.loc[cluster,"max"]:.1f}')
plt.xlabel('孕周 (周)')
plt.ylabel('失败概率 (1 - 检测成功率)')
plt.title('不同BMI分组随孕周的检测成功概率')
plt.legend()
plt.savefig('km_style_curves.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='GC偏差', y='X染色体Z值绝对值', hue='检测成功', data=data_unique)
plt.xlabel('GC含量偏离50%的绝对值')
plt.ylabel('X染色体Z值绝对值')
plt.title('GC偏差对检测成功的影响')
plt.savefig('gc_deviation_scatter.png')
plt.close()

data_unique['GC偏差高'] = data_unique['GC偏差'] > 0.05
gc_groups = data_unique.groupby(['BMI分组', 'GC偏差高'])['检测成功'].mean().unstack()
print("按GC偏差和BMI分组的检测成功率：\n", gc_groups)

success_data = data_unique[data_unique['检测成功'] == 1]
abnormal_rate = success_data.groupby('BMI分组')['异常'].mean()
print("\n检测成功样本的异常率：\n", abnormal_rate)

print("\n判定规则：")
print("1. 若X染色体Z值绝对值 < 3，则判定检测成功。")
print("2. 对检测成功的样本，若13/18/21号染色体Z值绝对值 > 3，则判定为异常。")
print("3. 根据BMI分组中位孕周推荐最佳检测孕周。")

results_df = bmi_ranges.copy()
results_df['中位孕周'] = data_unique.groupby('BMI分组')['GA'].median().values
results_df['异常率'] = abnormal_rate.reindex(results_df['BMI分组']).values
results_df.to_csv('nipt_fourth_results.csv', index=False)
print("\n最终结果表格：\n", results_df)
