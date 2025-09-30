import datetime
import statistics
import matplotlib.pyplot as plt
import numpy as np

# Function to parse date and value from input string
def parse_input(input_str):
    parts = input_str.split(':')
    date_str, value_str = parts[0].strip(), parts[1].strip()
    date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
    value = float(value_str)
    return date, value

# Function to compute mean and standard deviation for each year
def compute_yearly_stats(data):
    yearly_data = {}
    for date, value in data:
        year = date.year
        if year not in yearly_data:
            yearly_data[year] = []
        yearly_data[year].append(value)
    yearly_stats = {}
    for year, values in yearly_data.items():
        if len(values) >= 2:
           yearly_stats[year] = {
            'mean': statistics.mean(values),
            'std_dev': statistics.stdev(values)
        }
        else:
            yearly_stats[year] = {
                'mean': statistics.mean(values),
                'std_dev': 0
            }

    return yearly_stats

# Function to plot time series graph
def plot_time_series(data):
    dates = [date for date, _ in data]
    values = [value for _, value in data]
    plt.plot(dates, values, label='Values', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Time Series Plot')
    plt.legend()
    plt.show()

# Function to plot moving average line
def plot_moving_average(data, window_size):
    dates = [date for date, _ in data]
    values = [value for _, value in data]
    moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    plt.plot(dates[window_size-1:], moving_avg, label='Moving Average', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Moving Average')
    plt.title('Moving Average Line')
    plt.legend()
    plt.show()

# Main program
data = []
while True:
    user_input = input("Enter 'YYYY-MM-DD: value' (or 'end' to finish input): ")
    if user_input == 'end':
        break
    date, value = parse_input(user_input)
    data.append((date, value))

yearly_stats = compute_yearly_stats(data)
print("Yearly statistics:")
for year, stats in yearly_stats.items():
    print(f"Year {year}: Mean = {stats['mean']}, Std Dev = {stats['std_dev']}")

plot_time_series(data)

# Optionally, plot moving average line
window_size = int(input("Enter moving average window size (0 to skip): "))
if window_size > 0:
    plot_moving_average(data, window_size)

plot_moving_average(data)
