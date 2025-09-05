# 稳定性与敏感性分析
# 剔除异常值（例如 Y 染色体浓度 < 0.01）
data_robust = data[data['Y染色体浓度'] > 0.01]
model_m3_robust = MixedLM.from_formula(
    "Y_concentration_logit ~ GA + 孕妇BMI + GA_BMI_interaction",
    data_robust,
    groups=data_robust['孕妇代码']
)
result_m3_robust = model_m3_robust.fit()
print("模型 M3（健壮性分析，剔除 Y 浓度 < 0.01 的样本）：")
print(result_m3_robust.summary())