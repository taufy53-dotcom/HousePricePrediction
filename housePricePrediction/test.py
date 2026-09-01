import pandas as pd
from sklearn.linear_model import Ridge


# -----------------------------------------
# Load dataset
# -----------------------------------------

df = pd.read_csv("data/house_dataset.csv")


# -----------------------------------------
# Preprocessing
# -----------------------------------------

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


# -----------------------------------------
# Train tuned Linear Regression model
# -----------------------------------------


# Decided on previous test that this model r2 score is 0.65 which is current best.
model = Ridge(alpha=0.01)

model.fit(X, y)


# -----------------------------------------
# Ask user for house information
# -----------------------------------------

while True:
    try:
        print("\n==============================")
        print("       HOUSE PRICE PREDICTOR")
        print("==============================")

        # Numerical inputs
        area = float(input("Enter area (sq ft): "))
        bedrooms = int(input("Enter number of bedrooms: "))
        bathrooms = int(input("Enter number of bathrooms: "))
        stories = int(input("Enter number of stories: "))
        parking = int(input("Enter number of parking spaces: "))

        # Check that numbers are valid
        if area <= 0:
            raise ValueError("Area must be greater than 0.")

        if bedrooms <= 0:
            raise ValueError("Bedrooms must be greater than 0.")

        if bathrooms <= 0:
            raise ValueError("Bathrooms must be greater than 0.")

        if stories <= 0:
            raise ValueError("Stories must be greater than 0.")

        if parking < 0:
            raise ValueError("Parking spaces cannot be negative.")

        # Yes/No inputs
        mainroad = input(
            "Is there a main road? (yes/no): "
        ).lower()

        guestroom = input(
            "Is there a guest room? (yes/no): "
        ).lower()

        basement = input(
            "Is there a basement? (yes/no): "
        ).lower()

        hotwaterheating = input(
            "Hot water heating? (yes/no): "
        ).lower()

        airconditioning = input(
            "Air conditioning? (yes/no): "
        ).lower()

        prefarea = input(
            "Is it in a preferred area? (yes/no): "
        ).lower()

        # Validate yes/no inputs
        yes_no_values = [
            mainroad,
            guestroom,
            basement,
            hotwaterheating,
            airconditioning,
            prefarea
        ]

        if not all(value in ["yes", "no"] for value in yes_no_values):
            raise ValueError("Please enter only 'yes' or 'no'.")

        # Furnishing status
        furnishingstatus = input(
            "Furnishing status (furnished/semi-furnished/unfurnished): "
        ).lower()

        if furnishingstatus not in [
            "furnished",
            "semi-furnished",
            "unfurnished"
        ]:
            raise ValueError(
                "Please enter furnished, semi-furnished, or unfurnished."
            )

        # If everything is correct, leave the loop
        break

    except ValueError as e:
        print("\n❌ Invalid input:", e)
        print("Please enter the information again.\n")

# -----------------------------------------
# Convert Yes/No values
# -----------------------------------------

input_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'mainroad': [mainroad],
    'guestroom': [guestroom],
    'basement': [basement],
    'hotwaterheating': [hotwaterheating],
    'airconditioning': [airconditioning],
    'parking': [parking],
    'prefarea': [prefarea],
    'furnishingstatus': [furnishingstatus]
})


for col in binary_columns:
    input_data[col] = input_data[col].map({
        'yes': 1,
        'no': 0
    })


# -----------------------------------------
# One-hot encode furnishing status
# -----------------------------------------

input_data = pd.get_dummies(
    input_data,
    columns=['furnishingstatus'],
    drop_first=True
)


# Make sure input has exactly the same columns
# as the training data
input_data = input_data.reindex(
    columns=X.columns,
    fill_value=0
)


# -----------------------------------------
# Make prediction
# -----------------------------------------

predicted_price = model.predict(input_data)[0]


# -----------------------------------------
# Display result
# -----------------------------------------

print("\n==============================")
print("       PREDICTION")
print("==============================")

print(
    f"Predicted House Price: ₹{predicted_price:,.2f}"
)

print("==============================\n")