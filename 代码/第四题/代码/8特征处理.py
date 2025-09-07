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