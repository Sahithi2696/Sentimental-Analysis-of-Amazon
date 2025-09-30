import datetime
import statistics
import matplotlib.pyplot as plt

# Initialize empty dictionaries to store values for each year
yearly_data = {}
yearly_mean = {}
yearly_std_dev = {}

# Initialize empty lists to store all values and dates
dates = []
values = []

# Read inputs until 'end' is entered
while True:
    user_input = input("Enter date and value (YYYY-MM-DD: value), or 'end' to finish: ")
    if user_input == 'end':
        break
    else:
        try:
            date_str, value = user_input.split(': ')
            date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            dates.append(date)
            values.append(float(value))
        except ValueError:
            print("Invalid input format. Please enter date and value in the format 'YYYY-MM-DD: value'.")

# Group values by year
for date, value in zip(dates, values):
    year = date.year
    if year not in yearly_data:
        yearly_data[year] = []
    yearly_data[year].append(value)

# Compute mean and standard deviation for each year
for year, data in yearly_data.items():
    yearly_mean[year] = statistics.mean(data)
    if len(data) >= 2:
        yearly_std_dev[year] = statistics.stdev(data)
    else:
        yearly_std_dev[year] = 0

sorted_dates, sorted_values = zip(*sorted(zip(dates, values)))

# Plot time series graph

def plot_time_series(sorted_dates,sorted_values):
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_dates, sorted_dates, sorted_values = zip(*sorted(zip(dates, values)))
, label='Values', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Time Series Plot of Values')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot moving average line
window_size = 3  # Adjust window size as desired
moving_average = [sum(values[i:i+window_size])/window_size for i in range(len(values)-window_size+1)]
moving_average = [None]*(window_size-1) + moving_average  # Pad with None to match original length
plt.plot(dates, moving_average, label='Moving Average', color='red', linestyle='--')
plt.legend()
plt.show()

# Print mean and standard deviation for each year
for year in sorted(yearly_mean.keys()):
    print(f"Year: {year}, Mean: {yearly_mean[year]}, Standard Deviation: {yearly_std_dev[year]}")


plot_time_series(sorted_dates,sorted_values)