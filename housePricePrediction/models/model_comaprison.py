import pandas as pd
import matplotlib.pyplot as plt


# Read result files
linear = pd.read_csv("../data/linear_results.csv")
decision_tree = pd.read_csv("../data/decision_tree_results.csv")


# Model names
models = ["Linear Regression", "Decision Tree"]


# Get values from CSV files
mse_scores = [
    linear["MSE"].iloc[0],
    decision_tree["MSE"].iloc[0]
]

rmse_scores = [
    linear["RMSE"].iloc[0],
    decision_tree["RMSE"].iloc[0]
]

r2_scores = [
    linear["R2 Score"].iloc[0],
    decision_tree["R2 Score"].iloc[0]
]


# Print comparison
print("\nMODEL COMPARISON")
print("======================")

for i in range(len(models)):
    print("\n", models[i])
    print("MSE :", mse_scores[i])
    print("RMSE:", rmse_scores[i])
    print("R²  :", r2_scores[i])


# ==================================================
# MSE COMPARISON
# ==================================================

plt.figure(figsize=(8, 5))

plt.bar(models, mse_scores)

plt.xlabel("Model")
plt.ylabel("Mean Squared Error")
plt.title("MSE Comparison")

for i, value in enumerate(mse_scores):
    plt.text(i, value, f"{value:,.0f}",
             ha="center", va="bottom")

plt.show()


# ==================================================
# RMSE COMPARISON
# ==================================================

plt.figure(figsize=(8, 5))

plt.bar(models, rmse_scores)

plt.xlabel("Model")
plt.ylabel("Root Mean Squared Error")
plt.title("RMSE Comparison")

for i, value in enumerate(rmse_scores):
    plt.text(i, value, f"{value:,.0f}",
             ha="center", va="bottom")

plt.show()


# ==================================================
# R² COMPARISON
# ==================================================

plt.figure(figsize=(8, 5))

plt.bar(models, r2_scores)

plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("R² Score Comparison")

plt.ylim(0, 1)

for i, value in enumerate(r2_scores):
    plt.text(i, value + 0.02, f"{value:.3f}",
             ha="center")

plt.show()