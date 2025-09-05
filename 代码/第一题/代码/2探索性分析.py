plt.figure(figsize=(10, 6))
for id in data['孕妇代码'].unique()[:10]:  
    subset = data[data['孕妇代码'] == id]
    plt.plot(subset['GA'], subset['Y染色体浓度'], marker='o', label=id)
plt.xlabel('孕周（周）')
plt.ylabel('Y 染色体浓度')
plt.title('部分孕妇 Y 染色体浓度变化轨迹')
plt.legend()
plt.show()


bmi_bins = [20, 28, 32, 36, 40, np.inf]
bmi_labels = ['[20,28)', '[28,32)', '[32,36)', '[36,40)', '>=40']
data['BMI_group'] = pd.cut(data['孕妇BMI'], bins=bmi_bins, labels=bmi_labels, right=False)

plt.figure(figsize=(10, 6))
sns.lineplot(x='GA', y='Y染色体浓度', hue='BMI_group', data=data)
plt.xlabel('孕周（周）')
plt.ylabel('Y 染色体浓度')
plt.title('不同 BMI 分组下的 Y 染色体浓度')
plt.show()


corr_vars = ['Y染色体浓度', 'GA', '孕妇BMI', '年龄', 'GC含量', '被过滤掉读段数的比例']
corr_matrix = data[corr_vars].corr(method='spearman')
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman 相关性矩阵')
plt.show()