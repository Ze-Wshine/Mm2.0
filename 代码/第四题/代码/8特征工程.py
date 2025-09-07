 # 特征工程
data['GC偏差'] = np.abs(data['GC含量'] - 0.5)
data['读段质量'] = data['唯一比对的读段数'] / data['原始读段数']
data['Z值综合'] = data[['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值']].abs().mean(axis=1)
data['检测成功'] = (data['X染色体的Z值'].abs() < 3).astype(int)

# 过滤数据
data = data[(data['GC含量'] >= 0.4) & (data['GC含量'] <= 0.6) & 
            (data['GA'] >= 10) & (data['GA'] <= 25)].copy()

# 取每位孕妇第一次检测数据
data_unique = data.sort_values(['孕妇代码', 'GA']).groupby('孕妇代码').first().reset_index()

# 异常标签（基于AB列）
data_unique['异常'] = data_unique['染色体的非整倍体'].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)