# Import libraries
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
script_dir = Path(__file__).resolve().parent
data_path = script_dir / "archive" / "train.csv"
data = pd.read_csv(data_path)

# Display first 5 rows
print(data.head())

# Drop Loan_ID column
data = data.drop("Loan_ID", axis=1)

# Separate features and target
X = data.drop("Credit_History", axis=1)
y = data["Credit_History"]

# Encode target variable
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Separate categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Fill missing numerical values
num_imputer = SimpleImputer(strategy='mean')
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

# Fill missing categorical values
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# Encode categorical variables
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
model_path = script_dir / "loan_model.pkl"
joblib.dump(model, str(model_path))

print(f"\nModel saved as {model_path}")