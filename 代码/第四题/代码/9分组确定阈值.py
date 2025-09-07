bins = [20, 25, 30, 35, 40, float('inf')]
labels = ['[20, 25)', '[25, 30)', '[30, 35)', '[35, 40)', '40+']
data_unique['BMI分组'] = pd.cut(data_unique['孕妇BMI'], bins=bins, labels=labels, right=False)

for label in labels:
    group_data = data_unique[data_unique['BMI分组'] == label].dropna(subset=['Z值综合', '异常'])
    print(f"BMI组 {label} 样本量: {len(group_data)}")

z_thresholds = {}
for label in labels:
    group_data = data_unique[data_unique['BMI分组'] == label].dropna(subset=['Z值综合', '异常'])
    if len(group_data) > 5:  # 要求至少5个样本
        # 初始阈值：均值 + 3倍标准差
        initial_threshold = group_data['Z值综合'].mean() + 3 * group_data['Z值综合'].std()
        fpr, tpr, thresholds = roc_curve(group_data['异常'], group_data['Z值综合'])
        youden_idx = np.argmax(tpr - fpr)
        z_thresholds[label] = max(min(thresholds[youden_idx], 5.0), 1.0)  # 限制范围1.0-5.0
    else:
        z_thresholds[label] = 3.0  # 默认值，若样本量不足

optimal_ga = data_unique.groupby('BMI分组')['GA'].median()
abnormal_rate = data_unique.groupby('BMI分组')['异常'].mean().fillna(0)  # 填充NaN为0

results_df = pd.DataFrame({
    'BMI范围': labels,
    'Z值阈值': [z_thresholds[label] for label in labels],
    '最佳孕周': optimal_ga,
    '异常率': abnormal_rate
})
results_df.to_csv('nipt_fourth_results.csv', index=False)

print("\n结果表格：")
print(results_df)