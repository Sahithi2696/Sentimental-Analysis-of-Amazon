# importing necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# reading the dataset and converting to dataframe
data = pd.read_csv('titanic.csv')
df = pd.DataFrame(data)
df.head()

# just checking the attributes
var = df.columns

# Check for missing values and then drops them
df.isnull().sum()
df.dropna(inplace=True)

# Encoding male as 0 and female as 1
df['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
df.head()

# train-test-split
X = df[['Age', 'Pclass', 'Sex', 'Fare']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Train random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf.predict(X_test)

# accuracy precision recall and f1 score based on the outcome
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))

conf_matrix = confusion_matrix(y_test, y_pred_rf)
print('Confusion matrix:\n', conf_matrix)

"""Comparing the performance of the random forest model with the decision tree modeltrained earlier), we can see that 
the random forest model generally performs better in terms of accuracy. This is because random forests are an 
ensemble method that combine multiple decision trees, each trained on a different subset of the data and use voting 
to make predictions. This helps to reduce overfitting and improve the model's accuracy. However, the random forest 
model may be more complex and computationally expensive than the logistic regression and decision tree models, 
so there may be a trade-off between model performance and computational resources."""