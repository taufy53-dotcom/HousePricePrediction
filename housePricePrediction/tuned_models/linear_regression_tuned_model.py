import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("../data/house_dataset.csv")

results = []

# Convert Yes/No columns
binary_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

for col in binary_columns:
    df[col] = df[col].map({
        'yes': 1,
        'no': 0
    })


# One-hot encode furnishing status
df = pd.get_dummies(
    df,
    columns=['furnishingstatus'],
    drop_first=True
)


# Features and target
X = df.drop('price', axis=1)
y = df['price']


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Different alpha values to test
alphas = [0.01, 0.1, 1, 10, 100, 1000]

# Store results for graph
r2_scores = []
rmse_scores = []

best_r2 = -float("inf")
best_alpha = None


print("Ridge Regression Hyperparameter Tuning")
print("---------------------------------------")


for alpha in alphas:

    model = Ridge(alpha=alpha)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Store calculated values
    r2_scores.append(r2)
    rmse_scores.append(rmse)

    print(
        "alpha:",
        alpha,
        "| RMSE:",
        round(rmse, 2),
        "| R²:",
        round(r2, 4)
    )

    # Store every result
    results.append({
        "Max Alpha": alpha,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })


    # Find best model
    if r2 > best_r2:
        best_r2 = r2
        best_alpha = alpha

# Save all tuning results to CSV
results_df = pd.DataFrame(results)

results_df.to_csv(
    "../data/linear_regression_tuning_results.csv",
    index=False
)


print("\nBest alpha:", best_alpha)
print("Best R²:", best_r2)


# ---------------------------------------
# Graph: Alpha vs R²
# ---------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    alphas,
    r2_scores,
    marker='o'
)

plt.xlabel("Alpha")
plt.ylabel("R² Score")
plt.title("Ridge Regression: Alpha vs R² Score")

plt.xscale("log")
plt.grid(True, alpha=0.3)

plt.show()


# ---------------------------------------
# Graph: Alpha vs RMSE
# ---------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    alphas,
    rmse_scores,
    marker='o'
)

plt.xlabel("Alpha")
plt.ylabel("RMSE")
plt.title("Ridge Regression: Alpha vs RMSE")

plt.xscale("log")
plt.grid(True, alpha=0.3)

plt.show()