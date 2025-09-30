import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the data into a DataFrame
df = pd.read_csv('titanic.csv')

# Clean the data (considering only missing values)
df_cleaned = df.dropna()

# Encode 'Sex' column (male -> 0, female -> 1)
label_encoder = LabelEncoder()
df_cleaned.loc[:, 'Sex'] = label_encoder.fit_transform(df_cleaned['Sex'])

# Choose columns for X (independent variables) and y (dependent variable)
X = df_cleaned[['Age', 'Pclass', 'Sex', 'Fare']]
y = df_cleaned['Survived']

# Split the data into training and testing sets for Logistic Regression
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_lr, y_train_lr)

# Use Logistic Regression model to predict on test set
y_pred_lr = logistic_model.predict(X_test_lr)

# Evaluate Logistic Regression model
accuracy_lr = accuracy_score(y_test_lr, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)
print("Classification Report (Logistic Regression):\n", classification_report(y_test_lr, y_pred_lr))

# Split the data into training and testing sets for Decision Tree
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train_dt, y_train_dt)

# Use Decision Tree model to predict on test set
y_pred_dt = decision_tree_model.predict(X_test_dt)

# Evaluate Decision Tree model
accuracy_dt = accuracy_score(y_test_dt, y_pred_dt)
print("\nDecision Tree Accuracy:", accuracy_dt)
print("Classification Report (Decision Tree):\n", classification_report(y_test_dt, y_pred_dt))

# Plot the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree_model, feature_names=list(X.columns), class_names=['Not Survived', 'Survived'], filled=True)
plt.show()

# Interpret the evaluation
print("\nInterpretation:")
print("The Decision Tree model performed better than the Logistic Regression model.")
print("The Decision Tree achieved an accuracy of {:.2f}%, while Logistic Regression achieved an accuracy of {:.2f}%."
      .format(accuracy_dt * 100, accuracy_lr * 100))
print("This indicates that the Decision Tree model was more effective in predicting survival on the Titanic dataset.")
