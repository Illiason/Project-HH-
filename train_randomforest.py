# Train Random Forest model to predict heart disease

# pandas = library for reading CSV files and working with tables
import pandas as pd
# numpy = library for working with numbers and arrays
import numpy as np
# train_test_split = function that splits data into training and testing sets
from sklearn.model_selection import train_test_split
# accuracy_score = calculates how many predictions were correct (percentage)
# classification_report = detailed report with precision, recall, f1-score
# confusion_matrix = table showing correct vs wrong predictions
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# RandomForestClassifier = Random Forest machine learning model for classification
from sklearn.ensemble import RandomForestClassifier

import matplotlib as plt

print("=" * 70)
print("TRAINING RandomForest MODEL - Heart Disease Prediction")


# 1. Load dataset
# Read the CSV file into a DataFrame (table with rows and columns)
df = pd.read_csv('main_dataset.csv')
# Print how many rows and columns the dataset has
print(f"\nDataset: {len(df)} rows, {len(df.columns)} columns")


# 2. Split features (X) and target (y)
# X = all columns EXCEPT 'target' (the 8 features we use to predict)
# drop('target', axis=1) = remove the 'target' column, axis=1 means column
X = df.drop('target', axis=1)
# y = only the 'target' column (what we want to predict: 0=no disease, 1=has disease)
y = df['target']

# Print the feature names
print(f"Features: {list(X.columns)}")
# Print how many patients have disease (1) and don't have disease (0)
print(f"Target: {y.value_counts().to_dict()}")

# 3. Split into training (80%) and testing (20%)
# X_train = features for training (734 rows)
# X_test = features for testing (184 rows)
# y_train = correct answers for training
# y_test = correct answers for testing (to check if model is right)
# test_size=0.2 = 20% goes to testing, 80% goes to training
# random_state=42 = same split every time we run (reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)
# Print how many rows in each set
print(f"\nTraining set: {len(X_train)} rows")
print(f"Testing set: {len(X_test)} rows")

# 4. Create and train RandomForest model
model = RandomForestClassifier(
    n_estimators=100,      # build 100 decision trees
    max_depth=5,           # each tree can be 5 levels deep
    random_state=123       # same results every time we run
)

print("\nTraining Random Forest model...")
# fit() = "learn from this data"
model.fit(X_train, y_train)
print("Training complete!")

# 5. Make predictions
# predict() = use what the model learned to predict new patients
# y_pred = list of predictions (0 or 1) for each test patient
y_pred = model.predict(X_test)

# 6. Evaluate model
# Compare predictions (y_pred) with actual answers (y_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("RESULTS")

# Print accuracy as decimal and percentage
print(f"\nAccuracy: {accuracy:.2f} ({accuracy*100:.2f}%)")

# Confusion Matrix - shows what the model got right and wrong
# cm[0][0] = correctly predicted no disease (True Negative)
# cm[0][1] = said disease but actually healthy (False Positive)
# cm[1][0] = said healthy but actually has disease (False Negative)
# cm[1][1] = correctly predicted has disease (True Positive)
cm = confusion_matrix(y_test, y_pred)

print(f"\nConfusion Matrix:")
print("                               Predicted")
print("                      No Disease  |  Has Disease")
print(f"  Actual No Disease:   {cm[0][0]:>5}      |      {cm[0][1]:>5}")
print(f"  Actual Has Disease:  {cm[1][0]:>5}      |      {cm[1][1]:>5}")

# Classification Report - detailed metrics for each class
# precision = of all patients predicted as "disease", how many actually have it?
# recall = of all patients who actually have disease, how many did we catch?
# f1-score = balance between precision and recall
# support = how many patients in each group
print(f"\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Has Disease']))


print("\n" + "=" * 70)
print("DONE!")

