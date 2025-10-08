"""
Sole Survivor Project - Simple Linear Regression Solution
--------------------------------------------------------
- Trains a regression model on past data to check how well initial ratings explain SurvivalScore.
- Reports basic metrics and creates simple plots.
- Predicts SurvivalScore for next season and prints the top 3 contestants.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# -----------------------------
# 1) Load data
# -----------------------------
DATA_PAST = Path("sole_survivor_past.csv")
DATA_NEXT = Path("sole_survivor_next.csv")

past = pd.read_csv(DATA_PAST)
next_season = pd.read_csv(DATA_NEXT)

# -----------------------------
# 2) Quick EDA / sanity checks
# -----------------------------
# Print shapes and head (kept brief)
print("Past shape:", past.shape)
print("Next shape:", next_season.shape)
print("\nColumns:", list(past.columns))

# Ensure expected columns exist
assert "SurvivalScore" in past.columns, "Training data must include SurvivalScore."

# -----------------------------
# 3) Prepare features/target
# -----------------------------
feature_cols = [c for c in past.columns if c not in ["Name", "SurvivalScore"]]
X = past[feature_cols].values
y = past["SurvivalScore"].values

X_next = next_season[feature_cols].values
names_next = next_season["Name"].values if "Name" in next_season.columns else np.arange(len(next_season))

# -----------------------------
# 4) Model: Standardize + Linear Regression
#    (Standardization makes coefficients more interpretable and model more stable)
# -----------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

# -----------------------------
# 5) Cross-validated R^2 to evaluate how well initial ratings explain SurvivalScore
# -----------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
print("\nCross-validated R^2 scores:", np.round(cv_r2, 3))
print("Mean CV R^2:", cv_r2.mean().round(3), "±", cv_r2.std().round(3))

# Also train/validate split for simple error metrics
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)
print("\nHoldout metrics:")
print("  R^2 :", r2_score(y_te, y_pred).round(3))
print("  MAE :", mean_absolute_error(y_te, y_pred).round(3))
print("  RMSE:", np.sqrt(mean_squared_error(y_te, y_pred)).round(3))

# -----------------------------
# 6) Simple plots (saved as PNGs)
# -----------------------------
# Scatter: True vs Pred (holdout)
plt.figure()
plt.scatter(y_te, y_pred)
plt.xlabel("Actual SurvivalScore")
plt.ylabel("Predicted SurvivalScore")
plt.title("Actual vs Predicted (Holdout)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_actual_vs_pred.png", dpi=150)
plt.close()

# Feature correlation with target (Pearson)
corr = past[feature_cols + ["SurvivalScore"]].corr()["SurvivalScore"].drop("SurvivalScore").sort_values(ascending=False)
plt.figure()
corr.plot(kind="bar")
plt.ylabel("Correlation with SurvivalScore")
plt.title("Feature–Target Correlations")
plt.tight_layout()
plt.savefig("plot_feature_correlations.png", dpi=150)
plt.close()

# -----------------------------
# 7) Fit on all past data and predict next season
# -----------------------------
model.fit(X, y)
pred_next = model.predict(X_next)

# Top 3 predictions
order = np.argsort(pred_next)[::-1]  # descending
top3_idx = order[:3]

print("\nTop 3 predicted contestants:")
for i, idx in enumerate(top3_idx, start=1):
    print(f"  {i}) {names_next[idx]}  ->  Predicted SurvivalScore: {pred_next[idx]:.2f}")

# Save predictions to CSV
out = pd.DataFrame({
    "Name": names_next,
    "PredictedSurvivalScore": pred_next
}).sort_values("PredictedSurvivalScore", ascending=False)

out.to_csv("sole_survivor_predictions.csv", index=False)
print("\nSaved predictions to: sole_survivor_predictions.csv")
print("Saved plots: plot_actual_vs_pred.png, plot_feature_correlations.png")
