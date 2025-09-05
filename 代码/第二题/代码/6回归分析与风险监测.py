X = data[['GA', '孕妇BMI']].dropna()
y = data['Y_reached'].dropna()
model_lr = LogisticRegression().fit(X, y)

def risk_function_updated(ga, bmi, model, w1=0.7, w2=0.3):
    X_pred = pd.DataFrame({'GA': [ga], '孕妇BMI': [bmi]})
    prob_false_negative = 1 - model.predict_proba(X_pred)[:, 1]
    delay_penalty = ((ga - 12) / 15) ** 2
    return w1 * prob_false_negative + w2 * delay_penalty

ga_grid = np.arange(10, 25, 0.1)

quantiles_80 = data_unique.groupby('BMI_group')['GA_first'].quantile(0.8).rename('NIPT_80_Quantile')
risk_results_updated = {}
for group in data_unique['BMI_group'].unique():
    bmi_mean = data_unique[data_unique['BMI_group'] == group]['孕妇BMI'].mean()
    risks = [risk_function_updated(ga, bmi_mean, model_lr) for ga in ga_grid]
    optimal_ga = ga_grid[np.argmin(risks)]
    risk_results_updated[group] = optimal_ga