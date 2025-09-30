import pandas as pd

# Function from Question 1 to process reviews and update the vocabulary
def process_review(review, vocabulary):
    # Remove punctuation symbols
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    review = ''.join([char for char in review if char not in punctuation])

    # Split review into words
    words = review.split()

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

# Create an empty DataFrame called new_df to store the word counts
new_df = pd.DataFrame()

# Apply the process_review function to each review and update the vocabulary
for review in df['text']:
    word_count = process_review(review, vocabulary)

    # Create a dictionary for the current review's word count
    row_data = {'rating': 0}  # Initialize with 0 for all words in vocabulary
    for word, count in word_count.items():
        if word in vocabulary:
            row_data[word] = count

    # Append the row to new_df
    new_df = pd.concat([new_df, pd.DataFrame([row_data])], ignore_index=True)

# Display the new DataFrame structure
print("New DataFrame 'new_df' with word counts:")
print(new_df.head())
