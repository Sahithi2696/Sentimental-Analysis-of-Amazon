import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the data into a DataFrame
df = pd.read_csv('boston.csv')

# Check for missing values
missing_values = df.isnull().sum()

# Strategy: Impute missing values with mean
df_cleaned = df.fillna(round(df.mean(), 2))

# Display the first few rows of the cleaned data
print("\nAfter handling missing values:")
print(df_cleaned.head())

# Save cleaned data to a new CSV file in the same directory as the script
df_cleaned.to_csv("Boston_cleaned_data.csv", index=False)

# Choose 'RM' as the independent variable (X)
X = df_cleaned[['rm']]
# Choose 'medv' as the dependent variable (y)
y = df_cleaned['medv']

# Split the data into training and testing sets for Linear Regression (Question 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

# Make predictions with Linear Regression
y_pred_linear = model_linear.predict(X_test)

# Evaluate Linear Regression performance
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Print Linear Regression results
print("\nResults for Linear Regression (Question 1):")
print("Mean Squared Error (MSE):", mse_linear, "; R^2 Score:", r2_linear)
print("Equation of the LINEAR Regression: y =", model_linear.coef_[0], "* X +", model_linear.intercept_)

# Create arrays to store MSE and R2 scores for each degree
degrees = [2, 3, 4, 5]
mse_scores = []
r2_scores = []

# Train polynomial regression models for degrees 2 to 5
for degree in degrees:
    # Fit polynomial regression
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X_train)

    # Train the polynomial regression model
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y_train)

    # Make predictions on the testing set
    X_poly_test = poly_features.transform(X_test)
    y_pred_poly = model_poly.predict(X_poly_test)

    # Calculate MSE and R2 scores
    mse = mean_squared_error(y_test, y_pred_poly)
    r2 = r2_score(y_test, y_pred_poly)

    # Append scores to lists
    mse_scores.append(mse)
    r2_scores.append(r2)

    # Get coefficients and intercept
    coef = model_poly.coef_
    intercept = model_poly.intercept_

    # Generate the equation
    equation_terms = []
    for i in range(degree + 1):
        if i == 0:
            equation_terms.append(f"{intercept:.2f}")
        elif i == 1:
            equation_terms.append(f"{coef[1][i]:.2f} * X")
        else:
            equation_terms.append(f"{coef[0][i]:.2f} * X^{i}")
    equation = " + ".join(equation_terms)

    # Print the equation for this degree
    print(f"\nEquation for Degree-{degree} Polynomial Regression:")
    print("y =", equation)

# Plot the MSE and R2 scores for each degree
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(degrees, mse_scores, marker='o', color='blue')
plt.xlabel('Degree')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Degree')

plt.subplot(1, 2, 2)
plt.plot(degrees, r2_scores, marker='o', color='green')
plt.xlabel('Degree')
plt.ylabel('R^2 Score')
plt.title('R^2 Score vs Degree')

plt.tight_layout()
plt.show()

# Explanation of overfitting
print("\nExplanation of Overfitting:")
print("Overfitting occurs when a model learns the training data too well, capturing noise and random fluctuations.")
print(
    "As the degree of the polynomial increases, the model becomes more complex and can fit the training data very closely.")
print("However, this can lead to poor generalization to new, unseen data.")
print(
    "In the plot of MSE vs Degree, we can observe that as the degree increases, the MSE decreases on the training data,")
print("but it may increase on the testing data, indicating overfitting.")
print(
    "Similarly, in the plot of R^2 Score vs Degree, we see that the R^2 score increases with degree on the training data,")
print("but it may decrease on the testing data, again indicating overfitting.")
print("The linear regression model (degree 1) has lower complexity and may generalize better to unseen data,")
print("as shown by its relatively stable performance across the training and testing data.")
