"""
Sole Survivor – Linear Regression (Simple or Multiple)
------------------------------------------------------
- Toggle FEATURE_MODE = "simple" or "multiple"
- Prints CV and holdout metrics
- Saves plots:
    1) Actual vs Predicted (always)
    2) Best-fit line (only for simple regression)
- Predicts next season, prints top 3, saves CSV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------- CONFIG -------------
FEATURE_MODE = "multiple"        # "simple" or "multiple"
SINGLE_FEATURE = "Leadership"    # used only when FEATURE_MODE == "simple"

# ------------- DATA LOADING -------------
def load_data(past_file="sole_survivor_past.csv", next_file="sole_survivor_next.csv"):
    past = pd.read_csv(past_file)
    next_season = pd.read_csv(next_file)
    print("Past shape:", past.shape)
    print("Next shape:", next_season.shape)
    print("\nColumns:", list(past.columns))
    assert "SurvivalScore" in past.columns, "Training data must include SurvivalScore."
    return past, next_season

# ------------- FEATURE SELECTION & ARRAYS -------------
def prepare_data(past: pd.DataFrame, next_season: pd.DataFrame):
    all_features = [c for c in past.columns if c not in ["Name", "SurvivalScore"]]

    if FEATURE_MODE == "simple":
        feature_cols = [SINGLE_FEATURE]
        print(f"Using single feature: {SINGLE_FEATURE}")
    else:
        feature_cols = all_features
        print("Using multiple features:", feature_cols)

    X = past[feature_cols].values
    y = past["SurvivalScore"].values
    X_next = next_season[feature_cols].values
    names_next = (
        next_season["Name"].values
        if "Name" in next_season.columns
        else np.arange(len(next_season))
    )
    return feature_cols, X, y, X_next, names_next

# ------------- MODEL -------------
def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])

def evaluate_model(model, X, y, seed=42):
    # Cross-validation R^2
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
    print("\nCross-validated R² scores:", np.round(cv_r2, 3))
    print("Mean CV R²:", round(cv_r2.mean(), 3), "±", round(cv_r2.std(), 3))

    # Holdout split for error metrics and plotting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nHoldout metrics:")
    print("  R²   :", round(r2_score(y_test, y_pred), 3))
    print("  MAE  :", round(mean_absolute_error(y_test, y_pred), 3))
    print("  RMSE :", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))

    return X_train, y_train, y_test, y_pred, model

# ------------- PLOTS -------------
def save_plots(past, feature_cols, y_test, y_pred, X_train=None, y_train=None, model=None):
    # (1) Actual vs Predicted (always)
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual SurvivalScore")
    plt.ylabel("Predicted SurvivalScore")
    plt.title("Actual vs Predicted (Holdout)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_actual_vs_pred.png", dpi=150)
    plt.close()

    # (2) Best-fit line for SIMPLE regression (feature vs response)
    if len(feature_cols) == 1 and X_train is not None and model is not None:
        x_col = X_train[:, 0].reshape(-1, 1)
        # Sort for a clean line
        order = np.argsort(x_col[:, 0])
        x_sorted = x_col[order]
        y_line = model.predict(x_sorted)

        plt.figure()
        plt.scatter(x_col, y_train, color="blue", alpha=0.7, label="Training Data")
        plt.plot(x_sorted, y_line, color="red", linewidth=2, label="Best Fit Line")
        plt.xlabel(feature_cols[0])
        plt.ylabel("SurvivalScore")
        plt.title(f"Simple Linear Regression: {feature_cols[0]} vs SurvivalScore")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("plot_simple_fit.png", dpi=150)
        plt.close()

    # Feature–Target Correlation (works for simple or multiple)
    corr = (
        past[feature_cols + ["SurvivalScore"]]
        .corr()["SurvivalScore"]
        .drop("SurvivalScore")
        .sort_values(ascending=False)
    )
    plt.figure()
    corr.plot(kind="bar")
    plt.ylabel("Correlation with SurvivalScore")
    plt.title("Feature–Target Correlations")
    plt.tight_layout()
    plt.savefig("plot_feature_correlations.png", dpi=150)
    plt.close()

    if len(feature_cols) == 1:
        print("Saved plots: plot_actual_vs_pred.png, plot_simple_fit.png, plot_feature_correlations.png")
    else:
        print("Saved plots: plot_actual_vs_pred.png, plot_feature_correlations.png")

# ------------- PREDICT & OUTPUT -------------
def predict_next(model, X, y, X_next, names_next):
    model.fit(X, y)  # fit on ALL training data
    preds = model.predict(X_next)
    order = np.argsort(preds)[::-1]

    print("\nTop 3 predicted contestants:")
    for i, idx in enumerate(order[:3], start=1):
        print(f"  {i}) {names_next[idx]}  ->  Predicted SurvivalScore: {preds[idx]:.2f}")

    out = pd.DataFrame({"Name": names_next, "PredictedSurvivalScore": preds})
    out.sort_values("PredictedSurvivalScore", ascending=False, inplace=True)
    out.to_csv("sole_survivor_predictions.csv", index=False)
    print("\nSaved predictions to: sole_survivor_predictions.csv")

# ------------- MAIN -------------
def main():
    past, next_season = load_data()
    feature_cols, X, y, X_next, names_next = prepare_data(past, next_season)
    model = build_model()

    X_train, y_train, y_test, y_pred, fitted_holdout_model = evaluate_model(model, X, y)
    save_plots(past, feature_cols, y_test, y_pred, X_train=X_train, y_train=y_train, model=fitted_holdout_model)
    predict_next(model, X, y, X_next, names_next)

if __name__ == "__main__":
    main()
