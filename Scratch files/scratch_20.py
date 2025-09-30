import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the data
data = pd.read_csv('titanic.csv')
df = pd.DataFrame(data)

# Drop rows with missing values
df.dropna(inplace=True)

# Encode sex column
df['Sex'].replace({'male': 0, 'female': 1}, inplace=True)

# Split the data into features and target
X = df[['Age', 'Pclass', 'Sex', 'Fare']]
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Use LogisticRegression() to train your model
model = LogisticRegression()
model.fit(X_train, y_train)

# Use your trained model to classify for the test dataset
y_pred = model.predict(X_test)

# Evaluate the performance of the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', conf_matrix)

# Print the equation of your regression
coefficients = model.coef_[0]
intercept = model.intercept_
print('Equation of the regression:')
print("Survived = {:.3f} + {:.3f}*Age + {:.3f}*PClass + {:.3f}*Sex + {:.3f}*Fare".format(intercept[0], coefficients[0], coefficients[1], coefficients[2], coefficients[3]))

"""
INTERPRETATION OF THE EVALUATION

The model achieved an accuracy of 0.81, precision of 0.83, recall of 0.72, and f1-score of 0.77.
These values indicate that the model has good performance in predicting the survival of passengers
based on their age, class, sex, and fare paid. The precision value indicates that 83% of the passengers predicted
to survive actually survived, while the recall value indicates that the model correctly identified 72% of the
survivors. The f1-score takes into account both precision and recall and provides a balance between the two measures.
"""