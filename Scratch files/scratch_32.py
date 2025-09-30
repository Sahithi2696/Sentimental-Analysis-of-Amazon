import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np

# Import the data from "review_prepared.csv" into a pandas DataFrame called 'data'
data = pd.read_csv("review_prepared.csv")

# Set 'rating' as the dependent variable y
y = data['rating']

# Set all other columns as independent variables X
X = data.drop(columns=['rating'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Train the Random Forest Classifier on the entire training dataset
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Use the trained Random Forest Classifier to predict on the test dataset
y_pred_test_rf = rf_classifier.predict(X_test)

# Calculate performance metrics for Random Forest Classifier
accuracy_rf = accuracy_score(y_test, y_pred_test_rf)
precision_rf = precision_score(y_test, y_pred_test_rf, average='weighted', zero_division='warn')
recall_rf = recall_score(y_test, y_pred_test_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_test_rf, average='weighted')

print("=== Random Forest Classifier Performance ===")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-score: {f1_rf:.4f}")

# Train the Decision Tree Classifier on the entire training dataset
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Use the trained Decision Tree Classifier to predict on the test dataset
y_pred_test_dt = dt_classifier.predict(X_test)

# Calculate performance metrics for Decision Tree Classifier
accuracy_dt = accuracy_score(y_test, y_pred_test_dt)
precision_dt = precision_score(y_test, y_pred_test_dt, average='weighted', zero_division=0)
recall_dt = recall_score(y_test, y_pred_test_dt, average='weighted', zero_division=0)
f1_dt = f1_score(y_test, y_pred_test_dt, average='weighted', zero_division=0)

print("=== Decision Tree Classifier Performance ===")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1-score: {f1_dt:.4f}")

# Model Comparison
print("\n=== Model Comparison ===")
print("Random Forest Classifier:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-score: {f1_rf:.4f}")

print("\nDecision Tree Classifier:")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1-score: {f1_dt:.4f}")

# # Define the classifiers to compare
# classifiers = {
#     'Random Forest': RandomForestClassifier(),
#     'Decision Tree': DecisionTreeClassifier()
# }
#
# # Dictionary to store results
# results = {}
#
# # Iterate over classifiers
# for name, clf in classifiers.items():
#     accuracies = []
#     for train_index, test_index in kf.split(X):
#         X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
#         y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
#
#         clf.fit(X_train_cv, y_train_cv)
#         y_pred = clf.predict(X_test_cv)
#         acc = accuracy_score(y_test_cv, y_pred)
#         accuracies.append(acc)
#
    # Calculate mean accuracy
    mean_acc = np.mean(accuracies)
    results[name] = mean_acc

# Find the best model
best_model = max(results, key=results.get)
best_accuracy = results[best_model]

# Print results
print("\n=== Model Performance (10-Fold Cross-Validation) ===")
for model, accuracy in results.items():
    print(f"{model}: Mean Accuracy = {accuracy:.4f}")

print(f"\nBest Model: {best_model}")
print(f"Best Model Mean Accuracy: {best_accuracy:.4f}")
