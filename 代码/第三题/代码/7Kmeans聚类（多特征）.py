features = ['孕妇BMI', 'GA_first', '身高', '体重', '年龄']
X_cluster = data_unique[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
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