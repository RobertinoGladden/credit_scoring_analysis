# Credit Risk Analysis and Prediction

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Data Loading and Exploration
df = pd.read_csv('credit_risk_dataset.csv')

# Data Preprocessing
# Handle missing values
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)

# Remove outliers (e.g., age > 100 or emp_length > 50)
df = df[(df['person_age'] <= 100) & (df['person_emp_length'] <= 50)]

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
# Using RandomForestClassifier for its robustness and ability to handle imbalanced datasets
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Model Evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Testing Predictions
# Create sample test cases
sample_data = pd.DataFrame({
    'person_age': [25, 40, 30],
    'person_income': [50000, 120000, 80000],
    'person_home_ownership': ['RENT', 'MORTGAGE', 'OWN'],
    'person_emp_length': [5.0, 10.0, 3.0],
    'loan_intent': ['EDUCATION', 'DEBTCONSOLIDATION', 'MEDICAL'],
    'loan_grade': ['B', 'A', 'C'],
    'loan_amnt': [10000, 20000, 15000],
    'loan_int_rate': [10.99, 7.9, 13.49],
    'loan_percent_income': [0.2, 0.17, 0.19],
    'cb_person_default_on_file': ['N', 'Y', 'N'],
    'cb_person_cred_hist_length': [3, 15, 7]
})

# Preprocess sample data
for col in categorical_cols:
    sample_data[col] = le.fit_transform(sample_data[col])

# Scale sample data
sample_data_scaled = scaler.transform(sample_data)

# Make predictions
predictions = model.predict(sample_data_scaled)

# Display results
print("Inference Results for Sample Data:")
for i, pred in enumerate(predictions):
    status = 'Gagal Bayar' if pred == 1 else 'Lunas'
    print(f"Sample {i+1}: Predicted Loan Status = {status}")