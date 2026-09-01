import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("../data/house_dataset.csv")

# to save data for comparison
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


# Try different tree depths
depths = [2, 3, 4, 5, 6, 8, 10, None]

#List to store the value of r2 & RMSE
r2_scores = []
rmse_scores = []

best_r2 = -float("inf")
best_depth = None

print("Decision Tree Hyperparameter Tuning")
print("-----------------------------------")

for depth in depths:

    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    #Storing values of r2 & RMSE for visual representation
    r2_scores.append(r2)
    rmse_scores.append(rmse)

    print(
        "max_depth:",
        depth,
        "| RMSE:",
        round(rmse, 2),
        "| R²:",
        round(r2, 4)
    )

    # Store every result
    results.append({
        "Max Depth": depth,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })


    if r2 > best_r2:
        best_r2 = r2
        best_depth = depth


# Save all tuning results to CSV
results_df = pd.DataFrame(results)

results_df.to_csv(
    "../data/decision_tree_tuning_results.csv",
    index=False
)

print("\nBest max_depth:", best_depth)
print("Best R²:", best_r2)

#Visual Representation

depth_labels = ["2" , "3" , "4" , "5" , "6" , "8" , "10" , "unlimited"]

# Max depth / r2

plt.figure(figsize=(8,5))

plt.plot(
    depth_labels,
    r2_scores,
    marker = "o"
    
)

plt.xlabel("Maximum Depth")
plt.ylabel("R² Score")
plt.title("Decision Tree: Max Depth vs R² Score")

plt.grid(True, alpha=0.3)

plt.show()

# Max depth / RMSE

plt.figure(figsize=(8, 5))

plt.plot(
    depth_labels,
    rmse_scores,
    marker='o'
)

plt.xlabel("Maximum Depth")
plt.ylabel("RMSE")
plt.title("Decision Tree: Max Depth vs RMSE")

plt.grid(True, alpha=0.3)

plt.show()
