import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the CSV file
df = pd.read_csv('sales_data.csv')

# handle missing values
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# handle wrong values
df = df[(df['Units Sold'] >= 0) & (df['Units Sold'] <= 300)]
df = df[(df['Sales Price'] >= 5) & (df['Sales Price'] <= 20)]

# format the wrong format data
df['Date'] = pd.to_datetime(df['Date'])

# remove duplicated rows
df.drop_duplicates(inplace=True)

# Save cleaned data to a new CSV file in the same directory as the script
df.to_csv("cleaned_sales_data.csv", index=False)

# Print cleaned data
print("\nCleaned data:")
print(df)

# calculate total revenue
df['Revenue'] = df['Units Sold'] * df['Sales Price']
total_revenue = df['Revenue'].sum()

# calculate total units sold by product
total_units_sold = df.groupby('Product')['Units Sold'].sum()

# calculate top-selling product by unit sold
top_selling_product = total_units_sold.idxmax()

# create histogram of unit sold
plt.hist(df['Units Sold'], bins=20, color='purple')
plt.title('Histogram of Units Sold')
plt.xlabel('Units Sold')
plt.ylabel('Frequency')
plt.show()

# create column chart for each type of product - unit sold
df.groupby('Product')['Units Sold'].sum().plot(kind='bar', color='orange')
plt.title('Total Units Sold by Product')
plt.xlabel('Product')
plt.ylabel('Units Sold')
plt.show()