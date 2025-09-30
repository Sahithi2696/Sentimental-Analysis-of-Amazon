import pandas as pd

# Read the CSV file
df = pd.read_csv("sales_data.csv")

# Check the data types and missing values
print("Data Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# Convert 'Units Sold' and 'Sales Price' to numeric
df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')
df['Sales Price'] = pd.to_numeric(df['Sales Price'], errors='coerce')

# Handle missing values
df['Units Sold'].fillna(df['Units Sold'].mean(), inplace=True)
df['Sales Price'].fillna(df['Sales Price'].mean(), inplace=True)

# Filter out rows with incorrect 'Units Sold'
df = df[(df['Units Sold'] >= 0) & (df['Units Sold'] <= 300)]

# Specify multiple date formats for 'Date' column
date_formats = ['%m/%d/%y', '%m/%d/%Y', '%d/%m/%y', '%d/%m/%Y']  # Add more formats as needed
for fmt in date_formats:
    df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
    if not df['Date'].hasnans:  # If all dates are successfully parsed, break the loop
        break

# Remove duplicated data rows
df.drop_duplicates(inplace=True)

# Save cleaned data to a new CSV file in the same directory as the script
df.to_csv("cleaned_sales_data.csv", index=False)

# Print cleaned data
print("\nCleaned data:")
print(df)
