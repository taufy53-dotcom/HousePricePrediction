# 🏠 House Price Prediction

A machine learning project that predicts house prices based on various property-related features. The project includes the complete machine learning workflow — from data preprocessing and exploratory analysis to regression model training, evaluation, and an **interactive desktop GUI for making house price predictions**.

The project uses **Python, Pandas, Scikit-learn, Ridge Regression, and CustomTkinter** to combine machine learning with a user-friendly application.

---

## 📌 Project Overview

**House Price Prediction** is a supervised machine learning regression project designed to estimate the price of a house using information about its property features.

The model learns relationships between features such as:

* 📐 Area
* 🛏️ Number of bedrooms
* 🛁 Number of bathrooms
* 🏢 Number of stories
* 🚗 Parking spaces
* 🛣️ Main road availability
* 🛋️ Guest room
* 🏠 Basement
* ♨️ Hot water heating
* ❄️ Air conditioning
* 📍 Preferred area
* 🪑 Furnishing status

After training, the model can use these features to estimate the price of a new property.

The project also includes a **desktop GUI application** that allows users to enter property information through an intuitive interface instead of using the command line.

---

## 🎯 Objectives

* Understand the fundamentals of regression-based machine learning.
* Work with and preprocess a real-world house price dataset.
* Explore relationships between property features and house prices.
* Perform feature preprocessing and encoding.
* Train and evaluate a regression model.
* Use **Ridge Regression** for house price prediction.
* Build a user-friendly desktop application around the trained model.
* Connect a machine learning model with a practical Python application.
* Gain hands-on experience with Python, Pandas, Scikit-learn, and CustomTkinter.

---

## 🛠️ Technologies Used

| Technology              | Purpose                             |
| ----------------------- | ----------------------------------- |
| 🐍 **Python**           | Core programming language           |
| 🐼 **Pandas**           | Data loading and preprocessing      |
| 🔢 **NumPy**            | Numerical operations                |
| 📊 **Matplotlib**       | Data visualization                  |
| 🤖 **Scikit-learn**     | Machine learning and model training |
| 📈 **Ridge Regression** | House price prediction              |
| 🖥️ **CustomTkinter**   | Desktop GUI                         |
| 📓 **Jupyter Notebook** | Data analysis and experimentation   |

---

## 🧠 Machine Learning Model

The project uses **Ridge Regression** as the final prediction model.

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.01)

model.fit(X, y)
```

The selected model achieved an **R² score of approximately 0.65** during model testing, making it the current best-performing model in this project.

> **Note:** Model performance may vary depending on the dataset, preprocessing, train/test split, and evaluation methodology.

---

## 🔄 Machine Learning Workflow

The project follows a complete machine learning workflow:

### 1. 📥 Data Collection

The house price dataset is loaded using Pandas.

```python
df = pd.read_csv("data/house_dataset.csv")
```

### 2. 🔍 Data Exploration

The dataset is examined to understand:

* Available features
* Data types
* Feature distributions
* Relationships between variables
* Target variable characteristics

### 3. 🧹 Data Preprocessing

Categorical binary features such as `yes` and `no` are converted into numerical values.

```text
yes → 1
no  → 0
```

The following columns are processed:

```text
mainroad
guestroom
basement
hotwaterheating
airconditioning
prefarea
```

### 4. 🪑 Categorical Encoding

The `furnishingstatus` feature is converted into numerical features using one-hot encoding.

```python
pd.get_dummies(
    df,
    columns=["furnishingstatus"],
    drop_first=True
)
```

### 5. 🎯 Feature Selection

The `price` column is used as the target variable, while the remaining processed columns are used as input features.

```python
X = df.drop("price", axis=1)
y = df["price"]
```

### 6. 🤖 Model Training

A Ridge Regression model is trained using the processed dataset.

```python
model = Ridge(alpha=0.01)

model.fit(X, y)
```

### 7. 📊 Model Evaluation

The trained model is evaluated using regression metrics, with the current best model achieving an approximate **R² score of 0.65**.

### 8. 🏠 Price Prediction

The trained model accepts new property information and produces an estimated house price.

---

# 🖥️ Desktop GUI Application

The project includes a graphical interface built with **CustomTkinter**.

Instead of entering information through the terminal, users can enter property details through the application.

### GUI Features

* 🏠 Modern dark-themed interface
* 📐 Property area input
* 🛏️ Bedroom input
* 🛁 Bathroom input
* 🏢 Stories input
* 🚗 Parking input
* 🛣️ Main road selection
* 🛋️ Guest room selection
* 🏠 Basement selection
* ♨️ Hot water heating selection
* ❄️ Air conditioning selection
* 📍 Preferred area selection
* 🪑 Furnishing status selection
* 💰 Instant house price prediction
* ⚠️ Input validation
* 📊 Model information display

### Prediction Flow

```text
User enters property details
          ↓
       GUI Form
          ↓
    Input Validation
          ↓
   Data Preprocessing
          ↓
    Ridge Regression
          ↓
   Predicted House Price
          ↓
      GUI Result
```

---

## 📂 Project Structure

```text
HousePricePrediction/
│
├── data/
│   └── house_dataset.csv
│
├── housePricePrediction/
│   ├── dataset/
│   ├── models/
│   ├── tuned_models/
│   └── other project files
│
├── app.py
├── test.py
├── requirements.txt
├── LICENSE
└── README.md
```

### Important Files

| File                     | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| `app.py`                 | Desktop GUI application                               |
| `test.py`                | Command-line prediction implementation                |
| `data/house_dataset.csv` | House price dataset                                   |
| `housePricePrediction/`  | Machine learning notebooks, models, and project files |
| `requirements.txt`       | Python dependencies                                   |
| `README.md`              | Project documentation                                 |

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/taufy53-dotcom/HousePricePrediction.git
```

Move into the project directory:

```bash
cd HousePricePrediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you haven't created `requirements.txt` yet, you can install the required libraries manually:

```bash
pip install pandas numpy matplotlib scikit-learn customtkinter
```

---

## 4. Run the GUI

Start the desktop application with:

```bash
python app.py
```

On macOS/Linux you can also use:

```bash
python3 app.py
```

The **House Price Predictor** window will open.

---

# 🧪 Command-Line Version

The original prediction implementation can also be run from the terminal:

```bash
python test.py
```

The program will ask for the property information interactively and display the predicted house price.

---

# 📊 Example Prediction

Example input:

```text
Area: 7420 sq ft
Bedrooms: 4
Bathrooms: 2
Stories: 3
Parking: 2

Main Road: Yes
Guest Room: No
Basement: No
Hot Water Heating: No
Air Conditioning: Yes
Preferred Area: Yes

Furnishing Status: Furnished
```

The model then processes these values and produces an estimated house price.

> The prediction is an estimate generated by the trained machine learning model and should not be considered an actual market valuation.

---

# 🔮 Future Improvements

Some possible improvements for future versions include:

* [ ] Save the trained model using `joblib`
* [ ] Separate model training from the GUI application
* [ ] Add a model performance dashboard
* [ ] Add prediction history
* [ ] Add house price visualizations
* [ ] Add CSV-based batch predictions
* [ ] Improve model performance with additional algorithms
* [ ] Compare multiple regression models
* [ ] Add feature importance analysis
* [ ] Package the GUI as a standalone `.exe` / `.app`
* [ ] Add light/dark theme switching
* [ ] Add responsive GUI layouts

---

# 📈 Learning Outcomes

Through this project, I gained practical experience in:

* Python programming
* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Categorical encoding
* Regression algorithms
* Ridge Regression
* Model evaluation
* Pandas and Scikit-learn
* Building machine learning prediction systems
* Connecting ML models with desktop applications
* Developing user-friendly Python GUIs

---

# 👨‍💻 Author

**Taufique**

GitHub:
https://github.com/taufy53-dotcom

---

## ⭐ If You Like This Project

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub!

---

## 📄 License

This project is available under the **MIT License**.

