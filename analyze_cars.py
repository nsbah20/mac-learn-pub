import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# R^2 (R squared) tells you how much of the variation in the target variable 
# is explained by your model (higher is better).

# --- Utility functions ---
def run_simple_linear_regression(df, predictor, response):
    #Run and plot simple linear regression on one predictor.
    X = df[[predictor]]
    y = df[response]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    print("Simple Linear Regression (no split):")
    print(f"Predictor: {predictor}, Response: {response}")
    print(f"R^2 (all data): {model.score(X, y):.3f}")

    # Plot
    plt.scatter(X, y, color="blue", label="True values")
    plt.plot(X, y_pred, color="red", linewidth=2, linestyle='dotted', label="Prediction line")
    plt.xlabel(predictor)
    plt.ylabel(response)
    plt.title(f"{predictor} vs {response} (Simple Linear Regression)")
    plt.legend()
    plt.show()

    return model


def run_multiple_linear_regression(df, response, drop_cols=None):
    #Run multiple regression with one-hot encoding for all categorical variables.
    if drop_cols is None:
        drop_cols = []

    # Prepare predictors and response
    X = df.drop(columns=drop_cols + [response])
    X = pd.get_dummies(X, drop_first=True)  # one-hot encode all categoricals
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nMultiple Linear Regression (with one-hot encoding):")
    print(f"Response: {response}")
    print(f"R^2 (train): {model.score(X_train, y_train):.3f}")
    print(f"R^2 (test): {r2_score(y_test, y_pred):.3f}")
    print("Number of features after encoding:", X.shape[1])

    # Plot predicted vs actual prices for the test set
    plt.scatter(y_test, y_pred, color="green", alpha=0.6)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Multiple Linear Regression: Actual vs Predicted Price")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
    plt.show()

    # Plot feature importances (absolute value of coefficients)
    coefs = pd.Series(model.coef_, index=X.columns)
    top_coefs = coefs.abs().sort_values(ascending=False).head(10)  # Top 10 most important features
    plt.figure(figsize=(8, 5))
    top_coefs.plot(kind='barh', color='teal')
    plt.xlabel("Absolute Coefficient Value")
    plt.title("Top 10 Feature Importances (Multiple Linear Regression)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return model, X.columns


### Main workflow
def main():
    # Load dataset
    df = pd.read_csv("cars.csv")

    # Simple regression: engine size vs price
    run_simple_linear_regression(df, predictor="enginesize", response="price")

    # Multiple regression: use all predictors (drop ID & CarName)
    run_multiple_linear_regression(
        df,
        response="price",
        drop_cols=["car_ID", "CarName"]
    )
#(car_ID is just a unique identifier for each row (like a serial number). 
            #It does not contain any information useful for predicting price)

#CarName is a text label (the name of the car). 
# As a string, it is not directly useful for regression, 
# and one-hot encoding it would create a huge number of unnecessary columns (Possible overfitting).

if __name__ == "__main__":
    main()
