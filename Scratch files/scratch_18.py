import pandas as pd
from pyexpat import model

# %matplotlib inline

"""# 1.Load the Dataframe"""

df = pd.read_csv('boston.csv')

print("Data missing before addressing:")
print(df.head())

# Drop the 'Unnamed: 0' column
df = df.drop('Unnamed: 0', axis=1)

print("\nMissing values:")
print(df.shape)

"""# 2. Cleaning the data (Dealing with the null values)"""

df = df.fillna(round(df.mean(), 3))

# Save cleaned data to a new CSV file
df.to_csv("Boston_cleaned_data.csv", index=False)

"""# 3. Choosing independent dependent variable."""

# Independent variable
X = df.drop('medv', axis=1)
# dependent variable
y = df['medv']

print(X.shape)
print(y.shape)

"""# 4. Splitting the dataset in training and testing set."""

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("\nSplitting data into Training and Testing sets using ALL variables:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", Y_train.shape)
print("y_test shape:", Y_test.shape)

# Print the first few rows of the training and testing sets
print("\nTraining set:")
print("X_train:")
print(X_train.head())
print("\ny_train:")
print(Y_train.head())

print("\nTesting set:")
print("X_test:")
print(X_test.head())
print("\ny_test:")
print(Y_test.head())

"""# 5. Applying linear Regression"""

from sklearn import linear_model

model_2 = linear_model.LinearRegression()
model_2.fit(X_train, Y_train)

"""# 6. Prediction for test data"""

y_pred = model_2.predict(X_test)

"""# 7. Evaluation of the model"""

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_pred, Y_test)

# R_square
R_square = model_2.score(X_test, Y_test)
print('\nThe Mean Square Error(MSE) or J(theta) is: ', mse)
print('\nR square obtain for scikit learn library is :', R_square)

print('\nCoefficients:\n ', model_2.coef_)

"""# 8. Equation of the regression model for SINGLE vs ALL variables.


"""

# Print the equation of the regression for ALL Variables
print("\nEquation: Y =", end=" ")

# Print the coefficients
for i, coef in enumerate(model_2.coef_):
    print(f"{coef:.2f} * {X.columns[i]}", end=" + ")
print(f"{model_2.intercept_:.2f}")

from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Equation for the model using single independent variable 'rm'
print('\nCoefficients: ', model.coef_)
print("Intercept:", model.intercept_)

# Print the equation of the regression
print("\nEquation of the regression:")
print("y_pred = {:.8f} + ({:.8f})*X".format(model.intercept_[0], model.coef_[0][0]))


"""# 9.Interpreting the evaluation

In this case we have a much greater value of R square from the previous one that is 78% 
(Which denotes the accuracy of the model), which is far better than the first model. Also the MSE is very 
low from the first case. Conclusion is that increasing the number of features increase the accuracy of the model."""