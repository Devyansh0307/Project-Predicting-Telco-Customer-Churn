import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# --- 1. Initial Inspection ---
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

# --- 2. Check for Missing Values ---
print("\nMissing values in each column:")
print(df.isnull().sum())


# --- 3. Data Cleaning ---

# The 'TotalCharges' column has missing values and is an 'object' type.
# Let's see the rows where TotalCharges is missing.
print("\nRows with missing TotalCharges:")
print(df[df['TotalCharges'].isnull()]) # This shows these are new customers with 0 tenure.

# Let's convert TotalCharges to a numeric type. Errors='coerce' will turn any problematic
# values (like empty spaces) into 'NaN' (Not a Number).
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Now, we can fill the few missing values. Since these customers are new (tenure=0),
# their total charges should be 0.
df['TotalCharges'].fillna(0, inplace=True)

# We can also drop the 'customerID' column as it's just an identifier and has no predictive power.
df.drop('customerID', axis=1, inplace=True)


# --- 4. Convert Categorical Data to Numerical ---

# Separate target variable
X = df.drop('Churn', axis=1)
y = df['Churn']

# Convert the target variable 'Churn' to 0s and 1s
y = y.apply(lambda x: 1 if x == 'Yes' else 0)

# Use pandas get_dummies to one-hot encode the categorical features
# This creates new columns for each category and assigns a 0 or 1.
X_encoded = pd.get_dummies(X, drop_first=True)

print("\nShape of data after one-hot encoding:")
print(X_encoded.shape)
print("\nFirst 5 rows of encoded data:")
print(X_encoded.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 5. Scale Numerical Features ---
# It's good practice to scale features so that they have a similar influence.
scaler = StandardScaler()
# We only scale the columns that were originally numeric
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])


# --- 6. Split Data into Training and Testing Sets ---
# We'll use 80% of the data to train the model and 20% to test it.
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# --- 7. Train the Random Forest Model ---
# random_state is used for reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 8. Make Predictions on the Test Set ---
y_pred = model.predict(X_test)

# --- 9. Evaluate the Model ---
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# --- 10. Feature Importance ---
# This is one of the most powerful features of tree-based models.
feature_importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
feature_importances = feature_importances.nlargest(10) # Get the top 10 features

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Top 10 Feature Importances for Predicting Churn")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()