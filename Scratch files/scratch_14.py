import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data into a DataFrame
df = pd.read_csv('boston.csv')

# Display the first few rows to understand the data
print("Before handling missing values:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Strategy 2: Impute missing values with mean
df_cleaned = df.fillna(round(df.mean(),3))

# Display the first few rows of the cleaned data
print("\nAfter handling missing values:")
print(df_cleaned.head())


# Save the cleaned data to a new CSV file
df_cleaned.to_csv('boston_cleaned.csv', index=False)

# Choose 'RM' as the independent variable (X)
X = df_cleaned[['age']]

# Choose 'medv' as the dependent variable (y)
y = df_cleaned['ptratio']

# Print the first few rows to verify
print("\nX (Independent Variable - 'age'):")
print(X.head())

print("\ny (Dependent Variable - 'ptratio'):")
print(y.head())

# # Save X and y to a new CSV file
# X.to_csv('X_data.csv', index=False)
# y.to_csv('y_data.csv', index=False)

# Check if 'RM' is in the columns
if 'age' in df_cleaned.columns:
    # Choose 'RM' as the independent variable (X)
    X = df_cleaned[['age']]  # Double brackets to keep it as a DataFrame
    # Choose 'ptratio' as the dependent variable (y)
    y = df_cleaned['ptratio']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print the shapes of the training and testing sets
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Use LinearRegression to train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate and print the metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    predictions = model.predict(X_test)

    # Create a DataFrame to compare actual vs predicted values
    results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    print("\nActual vs Predicted:")
    print(results.head())

    # Scatter plot of Actual vs Predicted values with different colors
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, color='blue', label='Actual vs Predicted', alpha=0.7)

    # Plot the regression line
    x_line = np.linspace(min(y_test), max(y_test), 100)
    y_line = model.coef_[0] * x_line + model.intercept_
    plt.plot(x_line, y_line, color='green', linewidth=2, label='Regression Line')

    # Plot perfect prediction line
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--',
             label='Perfect Prediction')

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Values with Regression Line")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print the equation of the regression
    print("\nEquation of the regression:")
    print("y = {:.2f}x + {:.2f}".format(model.coef_[0], model.intercept_))

    # Interpretation of the evaluation
    print("\nInterpretation:")
    print("Mean Squared Error (MSE):")
    print("The Mean Squared Error (MSE) measures the average squared difference between predicted and actual values.")
    print("A lower MSE indicates better model performance. Our MSE is {:.2f}.".format(mse))
    print("\nR-squared (R^2) Score:")
    print(
        "The R-squared (R^2) score represents the proportion of variance in the dependent variable (medv) that is predictable from the independent variable (RM).")
    print("It ranges from 0 to 1, where 1 indicates a perfect fit. Our R^2 score is {:.2f}.".format(r2))


else:
    print("'age' column not found in the DataFrame.")