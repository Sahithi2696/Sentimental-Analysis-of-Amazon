import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("Amazon_reviews.csv")
# Summary of the dataset
print("Dataset Summary:\nThe dataset is focused on sentiment analysis of user reviews."
      "It contains both textual data (cleaned reviews) and numerical data "
      "(sentiment labels, review lengths, and scores)."
      "It is a product based review.")
print("Source: [Amazon Review]")
print("Topic: [Sentiment Analysis of Amazon Product Reviews]")
print(f"Number of records: {len(data)}")
print(f"Attributes: {', '.join(data.columns)}")

# Preprocess the dataset
print("\nPreprocessing:")
print("Before preprocessing, an example of data record:")
print(data.head(1).to_string(index=False, header=False))

#After Preprocessing
print("\nAfter preprocessing, an example of data record:")
print(data.head(1))

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Basic plots # Set the figure size to match the dimensions of a standard page
plt.figure(figsize=(11.69, 8.27))  # A4 paper size in inches (297mm × 210mm)
# Plot the first subplot (Sentiment Distribution) on the left
plt.subplot(1, 2, 1)
data['sentiments'].value_counts().plot(kind='bar', title='Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
# Plot the second subplot (Review Length Distribution) on the right
plt.subplot(1, 2, 2)
data['cleaned_review_length'].plot(kind='hist', title='Review Length Distribution')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.tight_layout()  # Ensure tight layout to prevent overlapping
plt.show()  # Show the plots

# Assign X, y to Train/test
X = data['cleaned_review']
y = data['sentiments']

# Handle & Fill missing values based on review score
default_message = lambda score: 'The products performance  is as expected' if score in [5, 4, 3] \
    else 'Performance of product needs to be improved'
X = X.fillna(data['review_score'].apply(default_message))
y = y.fillna('')

#Train/test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nSplitting into Training & Testing Sets:")
# Print the shapes of the training and testing sets
print("Training set size - X:", X_train.shape, " y:", y_train.shape)
print("Testing set size - X:", X_test.shape, " y:", y_test.shape)

# Classification
print("\nClassification:")
print("Using Naive Bayes classifier for sentiment analysis.")
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)  # Train the model.
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)
y_pred = clf.predict(vectorizer.transform(X_test))  # Predict the model.

# Evaluation metrics # Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the Model:", accuracy)
print("Classification Report:")
# Set zero_division parameter
print(classification_report(y_test, y_pred, zero_division=1, labels=clf.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot the model
y_pred_proba = clf.predict_proba(vectorizer.transform(X_test))
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_proba[:, 0], alpha=0.5, label='Negative')
plt.scatter(y_test, y_pred_proba[:, 1], alpha=0.5, label='Positive')
plt.xlabel('True Label')
plt.ylabel('Predicted Probability')
plt.title('Naive Bayes Classifier')
plt.legend()
plt.show()
