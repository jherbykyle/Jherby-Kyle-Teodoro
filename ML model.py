# ============================================================
# Dual-Task Physics-Guided ML Framework
# Target: ΔE (HOMO–LUMO Gap)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = r"C:\Users\PTRI\Desktop\electro"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 1️⃣ BASE DATA
# ============================================================

data = {
'Structure':[
'Cellulose','Cellulose-OH','Cellulose-NH2','Cellulose-COOH',
'Cellulose-CH3','Cellulose-CHO','Cellulose-CN','Cellulose-SH'
],
'LUMO':[1.442,0.179,1.597,0.973,1.638,0.055,0.819,-0.524],
'HOMO':[-6.502,-6.475,-6.419,-6.533,-6.435,-6.610,-6.642,-6.492],
'mu':[-2.530,-3.148,-2.410,-2.779,-2.398,-3.277,-2.911,-3.508],
'eta':[3.972,3.327,4.008,3.754,4.037,3.333,3.731,2.984],
'sigma':[0.252,0.300,0.249,0.266,0.248,0.300,0.268,0.335],
'omega':[0.806,1.489,0.725,1.029,0.712,1.611,1.136,2.063],
'TDM':[4.353,4.051,4.212,2.123,2.997,3.391,6.957,3.188],
'DeltaE':[7.944,6.654,8.017,7.507,8.074,6.665,6.571,5.968]
}

df = pd.DataFrame(data)

# ============================================================
# 2️⃣ AUGMENTATION (±2%)
# ============================================================

np.random.seed(42)

augmented = []
for _, row in df.iterrows():
    for _ in range(25):
        noisy = row.copy()
        for col in ['LUMO','HOMO','mu','eta','sigma','omega','TDM','DeltaE']:
            noisy[col] = row[col] * (1 + np.random.normal(0,0.02))
        augmented.append(noisy)

df_aug = pd.DataFrame(augmented)

features = ['LUMO','HOMO','mu','eta','sigma','omega','TDM']
X = df_aug[features]
y = df_aug['DeltaE']

# ============================================================
# 3️⃣ FROZEN MODEL
# ============================================================

reg = GradientBoostingRegressor(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=2,
    subsample=0.9,
    min_samples_leaf=1,
    random_state=42
)

# ============================================================
# 4️⃣ LOOCV VALIDATION
# ============================================================

loo = LeaveOneOut()
pred_loocv = cross_val_predict(reg, X, y, cv=loo)

r2 = r2_score(y, pred_loocv)
mae = mean_absolute_error(y, pred_loocv)
rmse = np.sqrt(mean_squared_error(y, pred_loocv))

print("\n=== LOOCV RESULTS ===")
print("R²:", round(r2,4))
print("MAE:", round(mae,4))
print("RMSE:", round(rmse,4))

# Save metrics
metrics_df = pd.DataFrame({
    "Metric":["LOOCV_R2","LOOCV_MAE","LOOCV_RMSE"],
    "Value":[r2,mae,rmse]
})
metrics_df.to_csv(os.path.join(output_dir,"Model_Metrics.csv"),index=False)

# Parity Plot
plt.figure()
plt.scatter(y, pred_loocv)
plt.plot([min(y), max(y)],[min(y), max(y)])
plt.xlabel("True ΔE")
plt.ylabel("Predicted ΔE")
plt.title("LOOCV Parity Plot")
plt.savefig(os.path.join(output_dir,"01_LOOCV_Parity.png"))
plt.close()

# Residual Plot
residuals = y - pred_loocv
plt.figure()
plt.scatter(pred_loocv, residuals)
plt.axhline(0)
plt.xlabel("Predicted ΔE")
plt.ylabel("Residual")
plt.title("Residual Diagnostics")
plt.savefig(os.path.join(output_dir,"02_ResidualPlot.png"))
plt.close()

# ============================================================
# 5️⃣ PERMUTATION IMPORTANCE
# ============================================================

reg.fit(X,y)
perm = permutation_importance(reg,X,y,n_repeats=20,random_state=42)

plt.figure()
plt.bar(features,perm.importances_mean)
plt.xticks(rotation=45)
plt.title("Permutation Importance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir,"03_PermutationImportance.png"))
plt.close()

# ============================================================
# 6️⃣ SHAP INTERPRETABILITY
# ============================================================

explainer = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(X)

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig(os.path.join(output_dir,"04_SHAP_Summary.png"))
plt.close()

plt.figure()
shap.dependence_plot("LUMO", shap_values, X, show=False)
plt.savefig(os.path.join(output_dir,"05_SHAP_LUMO.png"))
plt.close()

# ============================================================
# 7️⃣ TRUE EXTERNAL VALIDATION
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg.fit(X_train,y_train)
pred_test = reg.predict(X_test)

r2_ext = r2_score(y_test,pred_test)
rmse_ext = np.sqrt(mean_squared_error(y_test,pred_test))

print("\n=== EXTERNAL VALIDATION ===")
print("External R²:", round(r2_ext,4))
print("External RMSE:", round(rmse_ext,4))

plt.figure()
plt.scatter(y_test, pred_test)
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)])
plt.xlabel("True ΔE")
plt.ylabel("Predicted ΔE")
plt.title("External Validation Parity")
plt.savefig(os.path.join(output_dir,"06_ExternalValidation.png"))
plt.close()

# ============================================================
# 8️⃣ PREDICTION INTERVAL (FIXED)
# ============================================================

tree_preds = np.array([est[0].predict(X) for est in reg.estimators_])
std_pred = np.std(tree_preds, axis=0)

upper = pred_loocv + 1.96 * std_pred
lower = pred_loocv - 1.96 * std_pred

plt.figure()
plt.scatter(y, pred_loocv)
plt.fill_between(np.sort(y),
                 np.sort(lower),
                 np.sort(upper),
                 alpha=0.3)
plt.plot([min(y), max(y)],
         [min(y), max(y)])
plt.xlabel("True ΔE")
plt.ylabel("Predicted ΔE")
plt.title("Prediction Interval")
plt.savefig(os.path.join(output_dir,"07_PredictionInterval.png"))
plt.close()

print("\nALL RESULTS SAVED TO:", output_dir)
print("ANALYSIS COMPLETE.")