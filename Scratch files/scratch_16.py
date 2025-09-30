import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %matplotlib inline

"""# 1.Load the Dataframe and displaying the fist 5 rows"""
df = pd.read_csv('boston.csv')

df.head(5)

df = df.drop('Unnamed: 0', axis=1)

df.head(5)

print(df.shape)

"""# 2. Cleaning the data (Dealing with the null values)"""

df.isnull().sum()

"""# Since their is only one or two null values, We can use the previous values(values from the prior column) to fill 
the null values """

df["crim"].fillna(method='ffill', limit=1, inplace=True)
df["nox"].fillna(method='ffill', limit=1, inplace=True)
df["rm"].fillna(method='ffill', limit=1, inplace=True)
df["age"].fillna(method='ffill', limit=1, inplace=True)
df["dis"].fillna(method='ffill', limit=1, inplace=True)
df["rad"].fillna(method='ffill', limit=1, inplace=True)
df["lstat"].fillna(method='ffill', limit=1, inplace=True)
df["medv"].fillna(method='ffill', limit=1, inplace=True)

df.isnull().sum()

"""Now their is no null values in any of the column

# 3. Choosing one independent and one dependent(medv) variable.
"""

X_1 = df['crim']
Y_1 = df['medv']
X = pd.DataFrame(X_1)
Y = pd.DataFrame(Y_1)

print(X.shape)
print(Y.shape)

"""# 4. Splitting the dataset in training and testing set."""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""# 5. Applying linear Regression"""

from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

"""# 6. Prediction for test data"""

y_pred = model.predict(X_test)

"""# 7. Evaluation of the model"""

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_pred, y_test)

# R_square
R_square = model.score(X_test, y_test)
print('The Mean Square Error(MSE) or J(theta) is: ', mse)
print('R square obtain for scikit learn library is :', R_square)

"""# 8. Equation of the regression model"""

print('Coefficients: ', model.coef_)
print("Intercept:", model.intercept_)

"""Since we know the values of the coefficient, We can generate the the Equation using the general equation Y = mX +c

So, The equation is : y_pred = 24.51621785 + (-0.42997629)* X
Where X is the testing value

# 9. Interpreting the evaluation

The Mean Squared Error (MSE), which can be used to evaluate the evaluation, is the average squared difference between 
the predicted and actual values of the dependent variable. Better performance is indicated by a lower MSE value. 
Indicating how well the model fits the data is the R-squared number. It displays the percentage of the dependent 
variable's variance that the independent variable explains (s). A score of 0 means that the model does not explain 
any of the variation in the dependent variable, whereas a value of 1 shows a perfect fit. The model is said to have a 
good accuracy if it have low MSE and a high R Square value. But in our case the situation is opposite, High MSE value 
and a Low R Square value. So we can conclude that our model is not working good.


# Extra Credit: 10. Plot to represent linear regression model"""

import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='mediumpurple', label='Actual')
plt.plot(X_test, y_pred, color='mediumturquoise', linewidth=2, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Plot the regression line x_line = np.linspace(min(y_test), max(y_test), 10) y_line = model.coef_[0] * x_line + model.intercept_ plt.plot(x_line, y_line, color='green', linewidth=2, label='Regression Line')