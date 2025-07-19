# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("../career_path_in_all_field.csv")

# 1. Select features and target
X = df.drop(columns=["Field", "Career"])  # Features
y = df["Field"]  # Target

# --- ADD THIS LINE HERE ---
print("\n--- Actual dtypes of X after dropping 'Field' and 'Career' ---")
print(X.dtypes)
print("----------------------------------------------------------")
# --- END ADDITION ---

# 2. Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate performance
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Save model and label encoder
joblib.dump(model, "career_field_model.pkl")
joblib.dump(label_encoder, "field_label_encoder.pkl")

print("\n✅ Model and label encoder saved successfully!")

# This block was placed incorrectly at the end in your previous input
# It needs to be *before* train_test_split if it's to affect X
# Identify categorical columns in X
categorical_cols_in_X = X.select_dtypes(include=['object', 'category']).columns

# Apply One-Hot Encoding to X if categorical columns exist
if len(categorical_cols_in_X) > 0:
    print(f"\nFound categorical features in X: {list(categorical_cols_in_X)}. Applying One-Hot Encoding...")
    X = pd.get_dummies(X, columns=categorical_cols_in_X, drop_first=True)
else:
    print("\nNo categorical features found in X. Proceeding with numerical data.")

# Now proceed with train_test_split and model training...