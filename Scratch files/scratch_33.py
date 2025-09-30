import pandas as pd
import string

# Function to process reviews and update the vocabulary
def process_review(review, vocabulary):
    # Remove punctuation symbols and split review into words
    words = review.translate(str.maketrans('', '', string.punctuation)).split()

    # Remove stop words and short words
    stop_words = ['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'are', 'was', 'were']
    words = [word for word in words if word.lower() not in stop_words and len(word) >= 3]

    # Update the vocabulary
    vocabulary.update(words)

    # Count the occurrence of each word
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    # Return the word count dictionary
    return word_count


# 1. Import the data from "review_cleaned.csv" into a pandas DataFrame called df
df = pd.read_csv("review_cleaned.csv")

# Create an empty set called vocabulary to hold the unique words from all the reviews
vocabulary = set()

# Iterate through each review from 'text' column in df and update the vocabulary
for review in df['text']:
    process_review(review, vocabulary)

# Convert vocabulary set to a list for DataFrame columns
vocabulary_list = list(vocabulary)

# Choose the first 10 elements/columns in the vocabulary set
first_10_vocabulary = vocabulary_list[:10]
print("First 10 elements/columns in vocabulary set:")
print(first_10_vocabulary)

# Create a new DataFrame called new_df with columns from the vocabulary set
new_df = pd.DataFrame(columns=vocabulary_list)

# Iterate through each review to create word count dictionaries
word_count_list = []
for index, review in df.iterrows():
    word_count = process_review(review['text'], vocabulary.copy())  # Pass a copy of the vocabulary
    word_count['rating'] = review['rating']  # Add 'rating' column
    word_count_list.append(word_count)

# Concatenate the word count dictionaries into a DataFrame
new_df = pd.concat([pd.DataFrame([count]) for count in word_count_list], ignore_index=True)

# Reorder columns to have 'rating' as the first column
new_df = new_df[['rating'] + [col for col in new_df.columns if col != 'rating']]

# Write new_df to a new CSV file named "review_prepared.csv"
new_df.to_csv('review_prepared.csv', index=False)

# Print only 10 rows
print("New DataFrame 'new_df' with word counts:")
print(new_df.head(10))
