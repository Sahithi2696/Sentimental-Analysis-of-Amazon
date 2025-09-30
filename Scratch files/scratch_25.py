def process_review(review):
    # Split the review into a list of words
    words = review.split()

    # Define stop words
    stop_words = ['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'are', 'was', 'were']

    # Join words that occur after a full stop to the word before the full stop
    joined_words = []
    prev_word = None
    for word in words:
        if prev_word and prev_word[-1] == '.':
            joined_words[-1] += word
        else:
            joined_words.append(word)
        prev_word = word

    # Remove words less than 3 characters and stop words
    filtered_words = [word.lower() for word in joined_words if (len(word) >= 3) and (word.lower() not in stop_words)]

    # Count the occurrence of each word
    word_count = {}
    for word in filtered_words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    return word_count


# Given review
review = input("Enter your review: ")
# Call the function and print the resulting dictionary
result = process_review(review)
print(result)
