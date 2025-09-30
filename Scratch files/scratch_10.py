from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_and_split_data(test_size=0.2, random_state=42):
    """
    Load the breast cancer dataset and split it into training and testing sets.

    Parameters:
    - test_size: float, optional (default=0.2)
        The proportion of the dataset to include in the test split.
    - random_state: int or RandomState, optional (default=42)
        Controls the shuffling applied to the data before applying the split.

    Returns:
    - X_train: array-like, shape (n_samples, n_features)
        Training data.
    - X_test: array-like, shape (n_samples, n_features)
        Testing data.
    - y_train: array-like, shape (n_samples,)
        Training target.
    - y_test: array-like, shape (n_samples,)
        Testing target.
    """

    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

# Usage
X_train, X_test, y_train, y_test = load_and_split_data()

# Check the shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)  # Correct shape for binary classification
print("y_test shape:", y_test.shape)  # Correct shape for binary classification
