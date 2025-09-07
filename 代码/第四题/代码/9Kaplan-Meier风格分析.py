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
