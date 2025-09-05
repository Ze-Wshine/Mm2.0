model_m1 = MixedLM.from_formula(
    "Y_concentration_logit ~ GA",
    data,
    groups=data['孕妇代码']
)
result_m1 = model_m1.fit()
print("模型 M1（仅孕周）：")
print(result_m1.summary())

model_m2 = MixedLM.from_formula(
    "Y_concentration_logit ~ GA + 孕妇BMI",
    data,
    groups=data['孕妇代码']
)
result_m2 = model_m2.fit()
print("模型 M2（孕周 + BMI）：")
print(result_m2.summary())

data['GA_BMI_interaction'] = data['GA'] * data['孕妇BMI']
model_m3 = MixedLM.from_formula(
    "Y_concentration_logit ~ GA + 孕妇BMI + GA_BMI_interaction",
    data,
    groups=data['孕妇代码']
)
result_m3 = model_m3.fit()
print("模型 M3（孕周 + BMI + 交互项）：")
print(result_m3.summary())

model_m4 = MixedLM.from_formula(
    "Y_concentration_logit ~ GA + 孕妇BMI + GA_BMI_interaction + 年龄 + GC含量 + IVF妊娠",
    data,
    groups=data['孕妇代码']
)
result_m4 = model_m4.fit()
print("模型 M4（完整模型，含协变量）：")
print(result_m4.summary())

gam = LinearGAM(s(0, n_splines=10) + s(1, n_splines=10)).fit(
    data[['GA', '孕妇BMI']], data['Y_concentration_logit']
)
print("GAM 模型结果：")
print(gam.summary())

data['predicted_m3'] = result_m3.fittedvalues
plt.figure(figsize=(10, 6))
sns.lineplot(x='GA', y='predicted_m3', hue='BMI_group', data=data)
plt.xlabel('孕周（周）')
plt.ylabel('预测的LogitY染色体浓度')
plt.title('不同 BMI 分组下的 LMM 预测轨迹')
plt.show()

XX, YY = np.meshgrid(np.linspace(data['GA'].min(), data['GA'].max(), 50),
                     np.linspace(data['孕妇BMI'].min(), data['孕妇BMI'].max(), 50))
Z = gam.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)

plt.figure(figsize=(10, 6))
contour = plt.contourf(XX, YY, Z, cmap='viridis')
plt.colorbar(contour, label='预测的LogitY染色体浓度')
plt.xlabel('孕周（周）')
plt.ylabel('BMI')
plt.title('GAM 模型预测曲面')
plt.show()