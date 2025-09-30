import pandas as pd

# 1. Import the data from "review.csv" into a pandas DataFrame called df
df = pd.read_csv("review.csv")

# Display the first few rows of the dataframe to check the data
print("Data of Original DataFrame:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()

# 2. Handle all missing values (empty, na, and ?) by filling missing values with appropriate values
df['rating'] = df['rating'].fillna(df['rating'].mean())
df['text'] = df['text'].fillna('')
df['studentType'] = df['studentType'].fillna(df['studentType'].mode()[0])
df['category'] = df['category'].fillna('Overall Experience')
df['reviewDate'] = pd.to_datetime(df['reviewDate'], format='%m/%d/%Y', errors='coerce')

# Display the first few rows of the cleaned dataframe
print("\nDataFrame after filling missing values:")
print(df.head())

# 3. Format the data
df['rating'] = df['rating'].astype(int)  # Convert 'rating' to integer type
df['text'] = df['text'].astype(str)      # Ensure 'text' is in string format
df['studentType'] = df['studentType'].str.capitalize()  # Capitalize 'studentType'
df['category'] = df['category'].str.strip().str.title()  # Title case for 'category'
# Format the date column
df['reviewDate'] = df['reviewDate'].dt.strftime('%m/%d/%Y')

# Reorder columns if needed
df = df[['rating', 'text', 'studentType', 'category', 'reviewDate']]

# Display the first few rows of the cleaned and formatted dataframe
print("\nDataFrame after formatting:")
print(df.head())

# 4. Identify and handle wrong values
# Check 'rating' column for values outside the range of 1-5
wrong_ratings = df[(df['rating'] < 1) | (df['rating'] > 5)]
if not wrong_ratings.empty:
    print("\nRows with wrong 'rating' values:")
    print(wrong_ratings)

    # Check 'rating' column for values outside the valid range (1-5)
    df.loc[df['rating'] < 1, 'rating'] = 1
    df.loc[df['rating'] > 5, 'rating'] = 5

    # Here you can choose to drop these rows or replace the wrong values with appropriate ones

# # Check 'studentType' and 'category' columns for any unexpected values
# unexpected_student_types = df[~df['studentType'].isin(['Freshman', 'Sophomore', 'Junior', 'Senior', 'Alum'])]
# unexpected_categories = df[~df['category'].isin(['Overall Experience', 'Online Learning'])]
#
# if not unexpected_student_types.empty:
#     print("\nRows with unexpected 'studentType' values:")
#     print(unexpected_student_types)
#
# if not unexpected_categories.empty:
#     print("\nRows with unexpected 'category' values:")
#     print(unexpected_categories)

# Handle any wrong values as needed (e.g., replace with appropriate values or drop rows)
# For example, if 'rating' is wrong, we can replace it with the mean
# df.loc[(df['rating'] < 1) | (df['rating'] > 5), 'rating'] = df['rating'].mean()

# Display the first few rows of the cleaned and formatted dataframe
print("\nDataFrame after handling wrong values:")
print(df.head())

# 5. Remove duplicated rows
df = df.drop_duplicates()

# 6. Remove all data before 1/1/2019
df['reviewDate'] = pd.to_datetime(df['reviewDate'], format='%m/%d/%Y', errors='coerce')
df = df[df['reviewDate'] >= '2019-01-01']

df.to_csv('review_cleaned.csv', index=False)
print("\nSuccessfully saved the cleaned DataFrame to 'review_cleaned.csv'")

