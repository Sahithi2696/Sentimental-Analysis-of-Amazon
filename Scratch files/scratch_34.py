import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


# Import the data from "review_prepared.csv" into a pandas DataFrame called 'data'
data = pd.read_csv("review_prepared.csv")

# Display the first few rows of the DataFrame to verify
print("First few rows of 'data' DataFrame:")
print(data.head())

# Set 'rating' as the dependent variable y
y = data['rating']

# Set all other columns as independent variables X
X = data.drop(columns=['rating'])

# Display the shapes of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

from sklearn.model_selection import train_test_split, KFold
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold = 1

# Lists to store the data
fold_data = []

for train_index, test_index in kf.split(X):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

    # Create DataFrames for the current fold
    fold_df = pd.DataFrame({
        'Fold': [fold],
        'X_train Shape': [X_train_cv.shape],
        'X_test Shape': [X_test_cv.shape],
        'y_train Shape': [y_train_cv.shape],
        'y_test Shape': [y_test_cv.shape]
    })

    fold_data.append(fold_df)
    fold += 1

# Concatenate the fold DataFrames
fold_df = pd.concat(fold_data, ignore_index=True)

# Print the table
print("=== 10-Fold Cross-Validation Sets ===")
print(fold_df)

# Train the RandomForestClassifier on the entire training dataset
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Use the trained model to predict on the test dataset
y_pred_test_rf = rf_classifier.predict(X_test)

# Calculate accuracy on the test dataset
accuracy_test_rf = accuracy_score(y_test, y_pred_test_rf)

print("=== Random Forest Classifier Performance ===")
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_test_rf)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred_test_rf, average='weighted', zero_division='warn')
recall = recall_score(y_test, y_pred_test_rf, average='weighted')
f1 = f1_score(y_test, y_pred_test_rf, average='weighted')

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test_rf)

print("=== RandomForestClassifier Performance ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Train the RandomForestClassifier on the entire training dataset
dt_classifier = RandomForestClassifier()
dt_classifier.fit(X_train, y_train)

# Use the trained model to predict on the entire dataset
y_pred_dt = dt_classifier.predict(X)

# You would typically use the test dataset for predictions
# y_pred_test_dt = dt_classifier.predict(X_test)
# Calculate accuracy on the entire dataset
accuracy_dt = accuracy_score(y, y_pred_dt)

# You would typically use the test dataset for evaluation
# accuracy_test_dt = accuracy_score(y_test, y_pred_test_dt)

# Calculate other metrics
precision_dt = precision_score(y, y_pred_dt, average='weighted', zero_division=0)
recall_dt = recall_score(y, y_pred_dt, average='weighted', zero_division=0)
f1_dt = f1_score(y, y_pred_dt, average='weighted', zero_division=0)

print("=== Decision Tree Classifier Performance ===")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1-score: {f1_dt:.4f}")

print("=== Model Comparison ===")
print("Random Forest Classifier:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nDecision Tree Classifier:")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1-score: {f1_dt:.4f}")


# Define the classifiers to compare
# classifiers = {
#     'Logistic Regression': LogisticRegression(),
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
#     # Calculate mean accuracy
#     mean_acc = np.mean(accuracies)
#     results[name] = mean_acc
#
# # Find the best model
# best_model = max(results, key=results.get)
# best_accuracy = results[best_model]
#
# # Print results
# print("=== Model Performance (10-Fold Cross-Validation) ===")
# for model, accuracy in results.items():
#     print(f"{model}: Mean Accuracy = {accuracy:.4f}")
#
# print(f"\nBest Model: {best_model}")
# print(f"Best Model Mean Accuracy: {best_accuracy:.4f}")
#
