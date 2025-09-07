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