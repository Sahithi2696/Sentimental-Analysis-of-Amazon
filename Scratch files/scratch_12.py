import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data into a DataFrame
df = pd.read_csv('boston.csv')

# Check for missing values
missing_values = df.isnull().sum()

# Strategy: Impute missing values with mean
df_cleaned = df.fillna(df.mean())

# Choose all independent variables as X
X = df_cleaned.drop('medv', axis=1)
# Choose 'medv' as the dependent variable (y)
y = df_cleaned['medv']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Print the first few rows of the training and testing sets
print("\nTraining set:")
print("X_train:")
print(X_train.head())
print("\ny_train:")
print(y_train.head())

print("\nTesting set:")
print("X_test:")
print(X_test.head())
print("\ny_test:")
print(y_test.head())

# Use LinearRegression to train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate and print the metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (All variables):", mse)
print("R^2 Score (All variables):", r2)

# Print the equation of the regression
print("\nEquation of the regression (All variables):")
equation = "y = {:.2f}".format(model.intercept_)
for i, coef in enumerate(model.coef_):
    equation += " + {:.2f} * {}".format(coef, X.columns[i])
print(equation)

# Interpretation of the evaluation
print("\nInterpretation:")
print("Mean Squared Error (MSE) (All variables):")
print("The Mean Squared Error (MSE) measures the average squared difference between predicted and actual values.")
print("A lower MSE indicates better model performance. Our MSE is {:.2f}.".format(mse))
print("\nR-squared (R^2) Score (All variables):")
print("The R-squared (R^2) score represents the proportion of variance in the dependent variable (medv) that is predictable from the independent variables.")
print("It ranges from 0 to 1, where 1 indicates a perfect fit. Our R^2 score is {:.2f}.".format(r2))

# Make a plot to represent the linear regression model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted', alpha=0.6)
plt.plot(y_test, y_test, color='red', label='Ideal', linewidth=2)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Linear Regression: Actual vs Predicted MEDV')
plt.legend()
plt.show()
