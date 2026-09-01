import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------
# Load tuning results
# ---------------------------------------

linear = pd.read_csv("../data/linear_regression_tuning_results.csv")
tree = pd.read_csv("../data/decision_tree_tuning_results.csv")


# ---------------------------------------
# Display the data
# ---------------------------------------

print("Linear Regression Tuning Results")
print("---------------------------------")
print(linear)

print("\nDecision Tree Tuning Results")
print("---------------------------------")
print(tree)


# ---------------------------------------
# Find best models
# ---------------------------------------

best_linear = linear.loc[linear["R2 Score"].idxmax()]
best_tree = tree.loc[tree["R2 Score"].idxmax()]


print("\nBest Linear Regression")
print("----------------------")
print("Alpha:", best_linear["Max Alpha"])
print("RMSE:", best_linear["RMSE"])
print("R²:", best_linear["R2 Score"])


print("\nBest Decision Tree")
print("------------------")
print("Max Depth:", best_tree["Max Depth"])
print("RMSE:", best_tree["RMSE"])
print("R²:", best_tree["R2 Score"])


# =====================================================
# GRAPH 1 - Linear Regression tuning
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    linear["Max Alpha"],
    linear["R2 Score"],
    marker="o"
)

plt.xscale("log")

plt.xlabel("Alpha")
plt.ylabel("R² Score")
plt.title("Linear Regression - Alpha Tuning")

plt.grid(True)

plt.show()


# =====================================================
# GRAPH 2 - Decision Tree tuning
# =====================================================

# Remove the None depth because it cannot be plotted
tree_plot = tree.dropna(subset=["Max Depth"])

plt.figure(figsize=(8, 5))

plt.plot(
    tree_plot["Max Depth"],
    tree_plot["R2 Score"],
    marker="o"
)

plt.xlabel("Max Depth")
plt.ylabel("R² Score")
plt.title("Decision Tree - Depth Tuning")

plt.grid(True)

plt.show()


# =====================================================
# GRAPH 3 - Best Model R² Comparison
# =====================================================

models = [
    "Linear Regression",
    "Decision Tree"
]

r2_scores = [
    best_linear["R2 Score"],
    best_tree["R2 Score"]
]

plt.figure(figsize=(8, 5))

bars = plt.bar(models, r2_scores)

plt.ylabel("R² Score")
plt.xlabel("Model")
plt.title("Best Tuned Models - R² Comparison")

plt.ylim(0, 1)

for bar, value in zip(bars, r2_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.4f}",
        ha="center"
    )

plt.show()


# =====================================================
# GRAPH 4 - Best Model RMSE Comparison
# =====================================================

rmse_scores = [
    best_linear["RMSE"],
    best_tree["RMSE"]
]

plt.figure(figsize=(8, 5))

bars = plt.bar(models, rmse_scores)

plt.ylabel("RMSE")
plt.xlabel("Model")
plt.title("Best Tuned Models - RMSE Comparison")

for bar, value in zip(bars, rmse_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:,.0f}",
        ha="center",
        va="bottom"
    )

plt.show()