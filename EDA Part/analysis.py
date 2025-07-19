# eda_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../career_path_in_all_field.csv")

# -------------------------------
# 1. Dataset Overview
# -------------------------------
print("Shape of the dataset:", df.shape)
print("\nData Types & Nulls:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# -------------------------------
# 2. Target Distribution (Field)
# -------------------------------
plt.figure(figsize=(12, 6))
field_counts = df['Field'].value_counts()
sns.barplot(x=field_counts.index, y=field_counts.values, palette="viridis")
plt.title("Distribution of Career Fields")
plt.xlabel("Field")
plt.ylabel("Number of Entries")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("field_distribution.png")
plt.show()

# -------------------------------
# 3. Feature Distributions
# -------------------------------
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(18, 14))
for idx, col in enumerate(numerical_cols):
    plt.subplot(4, 4, idx + 1)
    sns.histplot(df[col], kde=True, color='skyblue', bins=20)
    plt.title(col)
    plt.xlabel("")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("feature_distributions.png")
plt.show()

# -------------------------------
# 4. Correlation Matrix
# -------------------------------
plt.figure(figsize=(14, 12))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

# -------------------------------
# 5. Grouped Feature Analysis by Field
# -------------------------------
grouped_means = df.groupby('Field')[numerical_cols].mean()

plt.figure(figsize=(18, 10))
for idx, col in enumerate(numerical_cols):
    plt.subplot(4, 4, idx + 1)
    grouped_means[col].sort_values().plot(kind='bar', color='coral')
    plt.title(f"Avg {col} by Field")
    plt.ylabel("Mean Value")
    plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("grouped_feature_analysis_by_field.png")
plt.show()

# -------------------------------
# Done!
# -------------------------------
print("\n✅ EDA Complete. All visuals saved.")
