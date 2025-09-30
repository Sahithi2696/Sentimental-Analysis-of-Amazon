import pandas as pd

# Read the CSV file
df = pd.read_csv("sales_data.csv")

# Display the initial state of the DataFrame
print("Initial DataFrame:")
print(df)

# Convert 'Units Sold' and 'Sales Price' to numeric
df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')  # Convert to numeric
df['Sales Price'] = pd.to_numeric(df['Sales Price'], errors='coerce')  # Convert to numeric


# Handling wrong format (Convert Sales Price to float)
df['Sales Price'] = pd.to_numeric(df['Sales Price'], errors='coerce')

# Fill missing values with mean
df['Units Sold'] = df['Units Sold'].fillna(df['Units Sold'].mean())
df['Sales Price'] = df['Sales Price'].fillna(df['Sales Price'].mean())


# Convert 'Date' to datetime with specified format
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')


# Convert 'Sales Price' to float
df['Sales Price'] = pd.to_numeric(df['Sales Price'], errors='coerce')  # Convert to numeric

# Identify and handle wrong values
# Assuming 'Units Sold' should not be negative
df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')  # Convert to numeric
df['Units Sold'] = df['Units Sold'].apply(lambda x: max(0, x))

# Remove duplicated data rows
df.drop_duplicates(inplace=True)


# Display cleaned DataFrame
print("\nCleaned DataFrame:")
print(df)

# Write cleaned data back to CSV file
df.to_csv("cleaned_sales_data.csv", index=False)
print("\nCleaned data saved to 'cleaned_sales_data.csv'")


