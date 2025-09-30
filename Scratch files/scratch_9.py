from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_and_split_data(test_size=0.2, random_state=42):
    # Load the breast cancer dataset
    data = load_breast_cancer()

    # Split the data into features (X) and target (y)
    X = data.data
    y = data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

# Example usage:
X_train, X_test, y_train, y_test = load_and_split_data()

# Print the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
def load_and_split_data():
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Load and split the data
X_train, X_test, y_train, y_test = load_and_split_data()

# Initialize the logistic regression model with increased max_iter
model = LogisticRegression(random_state=42, max_iter=1000)  # Increase max_iter

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)