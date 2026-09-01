import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("../data/house_dataset.csv")

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
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-hot encode furnishing status
df = pd.get_dummies(
    df,
    columns=['furnishingstatus'],
    drop_first=True
)

model = LinearRegression()

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression")
print("-----------------------")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

"""
pd.DataFrame([{
    "Model": "Linear Regression",
    "MSE": mse,
    "RMSE": rmse,
    "R2 Score": r2
}]).to_csv("data/linear_results.csv", index=False)
"""

results = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred
})

print(results.head(10))

plt.scatter(y_test, y_pred)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

plt.show()