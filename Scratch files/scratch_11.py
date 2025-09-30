from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def breast_cancer_classification():
    # Step 1: Load the breast cancer dataset
    cancer_data = load_breast_cancer()
    X = cancer_data.data
    y = cancer_data.target

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train a logistic regression model on the training data
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # Step 4: Use the trained model to make predictions on the testing data
    y_pred = model.predict(X_test)

    # Step 5: Evaluate the model using accuracy, precision, recall, and F-1 metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Extra Credit: Use 10-fold cross-validation
    print("\nExtra Credit: 10-fold Cross-Validation")
    cv_scores = cross_val_score(model, X, y, cv=10)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Score:", cv_scores.mean())

    accuracy_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y, cv=10, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=10, scoring='recall')
    f1_scores = cross_val_score(model, X, y, cv=10, scoring='f1')
    print("Precision:", precision_scores.mean())
    print("Recall:", recall_scores.mean())
    print("F1 Score:", f1_scores.mean())

# Call the function to perform the classification task
breast_cancer_classification()