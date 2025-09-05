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
