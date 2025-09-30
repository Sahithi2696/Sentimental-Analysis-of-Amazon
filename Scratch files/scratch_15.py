import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data into a DataFrame
df = pd.read_csv('boston.csv')

# Check for missing values
missing_values = df.isnull().sum()

# Strategy: Impute missing values with mean
df_cleaned = df.fillna(df.mean())

# Choose 'RM' as the independent variable (X)
X = df_cleaned[['rm']]
# Choose 'medv' as the dependent variable (y)
y = df_cleaned['medv']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Polynomial Regression model with degree=2
poly_features = PolynomialFeatures(degree=2)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)

# Make predictions with Polynomial Regression
y_pred_poly = model_poly.predict(X_poly_test)

# Evaluate Polynomial Regression performance
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Print Polynomial Regression results
print("Results for Polynomial Regression (Degree=2) with 'RM' as Independent Variable:")
print("Mean Squared Error (MSE):", mse_poly)
print("R^2 Score:", r2_poly)
print("Equation of the regression:")
print("y =", model_poly.coef_[2], "* X^2 +", model_poly.coef_[1], "* X +", model_poly.intercept_)

# Interpretation of the evaluation
print("\nInterpretation:")
print("The Polynomial Regression model with degree=2 and 'RM' as the independent variable has:")
print("- A Mean Squared Error (MSE) of", mse_poly)
print("- An R^2 Score of", r2_poly)
print("- This means the model explains around", r2_poly*100, "% of the variance in the test data.")

# Extra credit: Plot the Polynomial Regression model
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred_poly, color='green', linewidth=2, label='Polynomial Regression (Degree=2)')
plt.xlabel('RM')
plt.ylabel('medv')
plt.title('Polynomial Regression (Degree=2) with RM as Independent Variable')
plt.legend()
plt.show()
