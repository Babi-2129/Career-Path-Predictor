import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Dataset
df = pd.read_csv("../career_path_in_all_field.csv")

# --- Data Preparation ---
# Define Features (X) and Target (y)
# 'Field' is your target. 'Career' is secondary, so drop it from features.
X = df.drop(['Field', 'Career'], axis=1)
y = df['Field']

# Check and handle any remaining non-numeric columns in X (Crucial for 0.07 accuracy)
# Assuming 'Extracurricular_Activities' through 'Industry_Certifications' are numerical
# based on your previous data snippet.
# If df.info() shows 'object' type for any of these, use pd.get_dummies like this:
# X = pd.get_dummies(X, columns=['Name_of_Your_Object_Column'], drop_first=True)


# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training & Hyperparameter Tuning (for best accuracy)
# Define the model
rf_model = RandomForestClassifier(random_state=42)

# Define parameters to search
param_grid = {
    'n_estimators': [50, 100, 200],      # Number of trees in the forest
    'max_features': ['sqrt', 'log2'],    # Number of features to consider at each split
    'max_depth': [10, 20, None],         # Maximum depth of the tree (None means unlimited)
    'min_samples_split': [2, 5, 10],     # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]        # Minimum number of samples required to be at a leaf node
}

# Setup GridSearchCV
print("Starting GridSearchCV for optimal hyperparameters...")
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
                           # cv=5: 5-fold cross-validation
                           # n_jobs=-1: Use all available CPU cores
                           # verbose=1: Show progress
                           # scoring='accuracy': Optimize for accuracy

grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
print(f"\nBest Hyperparameters: {grid_search.best_params_}")

# 4. Evaluate Best Model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy (with best params): {accuracy:.4f}") # Increased precision

# 5. Save the Best Model
joblib.dump(best_model, 'best_career_path_predictor_model.pkl')
print("Best model saved as 'best_career_path_predictor_model.pkl'")

# Optional: Print classification report for more details
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))