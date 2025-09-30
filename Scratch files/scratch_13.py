import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the data and convert to DataFrame
data = pd.read_csv('boston.csv')

# Step 2: Clean the data (handling missing values)
# In this case, we'll assume there are no missing values in 'RM' and 'MEDV'
# But you can handle missing values based on your dataset

# Step 3: Define X and y
X = data[['rm']]  # Independent variable: Average number of rooms
y = data['medv']  # Dependent variable: Median value of owner-occupied homes

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict on the test dataset
y_pred = model.predict(X_test)

# Step 7: Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Step 8: Print the equation of the regression line
print("Equation of the regression line:")
print("medv = {:.2f} * rm + {:.2f}".format(model.coef_[0], model.intercept_))

# Step 9: Interpret the evaluation metrics
# Mean Squared Error (MSE) measures the average squared difference between the predicted and actual values.
# R^2 Score measures the proportion of the variance in the dependent variable that is predictable from the independent variable.
# Higher R^2 indicates a better fit of the model to the data.

# Step 10: Plot the linear regression model
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Average number of rooms (rm)')
plt.ylabel('Median value of owner-occupied homes (medv)')
plt.title('Linear Regression: RM vs medv')
plt.legend()
plt.show()
