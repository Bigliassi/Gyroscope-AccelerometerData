import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Function to detect peaks and return their properties
def detect_significant_peaks(signal, height=None, distance=None, prominence=None, width=None):
    signal = signal.dropna()
    peaks, properties = find_peaks(signal, height=height, distance=distance, prominence=prominence, width=width)
    return peaks, properties

# Function to perform validity checks
def perform_validity_checks(df, expected_columns):
    # Check if all expected columns are present
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the data: {missing_columns}")
    
    # Check for NaN values in expected columns
    for col in expected_columns:
        if df[col].isnull().any():
            print(f"Warning: NaN values found in column '{col}'. They will be dropped for processing.")
    
    # Check if TimeStamp column is in datetime format
    if not np.issubdtype(df['TimeStamp'].dtype, np.datetime64):
        try:
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        except Exception as e:
            raise ValueError(f"TimeStamp column cannot be converted to datetime: {e}")
    
    return df

# Load the CSV file
file_path = input("Enter the path to the CSV file: ")
df = pd.read_csv(file_path, on_bad_lines='skip')

# Expected columns
expected_columns = ['TimeStamp', 'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

# Perform validity checks
try:
    df = perform_validity_checks(df, expected_columns)
except ValueError as e:
    print(f"Data validation error: {e}")
    exit(1)

# Drop rows with NaNs in any of the relevant columns
df_clean = df.dropna(subset=expected_columns)

# Extract cleaned data
timestamps = df_clean['TimeStamp'].reset_index(drop=True)
acc_data = df_clean[['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']].reset_index(drop=True)
gyro_data = df_clean[['Gyro_X', 'Gyro_Y', 'Gyro_Z']].reset_index(drop=True)

# Compute magnitude of accelerometer and gyroscope data
acc_magnitude = np.sqrt(acc_data['Accelerometer_X']**2 + acc_data['Accelerometer_Y']**2 + acc_data['Accelerometer_Z']**2)
gyro_magnitude = np.sqrt(gyro_data['Gyro_X']**2 + gyro_data['Gyro_Y']**2 + gyro_data['Gyro_Z']**2)

# Combine accelerometer and gyroscope magnitude data
combined_magnitude = acc_magnitude + gyro_magnitude

# Smooth the magnitude data using a rolling mean
window_size = 5  # Adjust window size as needed
combined_magnitude_smooth = combined_magnitude.rolling(window=window_size, center=True).mean()

# Compute adaptive threshold based on mean and standard deviation
mean = combined_magnitude_smooth.mean()
std = combined_magnitude_smooth.std()
threshold = mean + 1.0 * std  # Adjust multiplier as needed

# Detect peaks in the combined magnitude data
peaks, properties = detect_significant_peaks(
    combined_magnitude_smooth,
    height=threshold,
    prominence=std * 0.5  # Adjust multiplier as needed
)

# Now, filter out peaks that are too close together, keeping the first one
min_interval = pd.Timedelta(seconds=4)  # Minimum interval between peaks
filtered_peaks = []
last_peak_time = None

for idx in range(len(peaks)):
    current_peak = peaks[idx]
    current_time = timestamps.iloc[current_peak]
    
    if last_peak_time is None:
        # First peak
        filtered_peaks.append(current_peak)
        last_peak_time = current_time
    else:
        time_diff = current_time - last_peak_time
        if time_diff >= min_interval:
            # Enough time has passed since the last peak
            filtered_peaks.append(current_peak)
            last_peak_time = current_time
        else:
            # Peaks are too close; ignore this peak
            continue  # Skip to the next peak

# Convert filtered_peaks to numpy array for indexing
filtered_peaks = np.array(filtered_peaks)

# Plotting raw accelerometer and gyroscope data with detected peaks
plt.figure(figsize=(15, 7))

# Plot raw accelerometer data
for col in ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']:
    plt.plot(timestamps, df_clean[col], label=col)

# Plot raw gyroscope data
for col in ['Gyro_X', 'Gyro_Y', 'Gyro_Z']:
    plt.plot(timestamps, df_clean[col], label=col)

# Mark filtered peaks on the raw data
plt.plot(timestamps.iloc[filtered_peaks], combined_magnitude.iloc[filtered_peaks], 'ro', label='Significant Peaks')

plt.title("Raw Accelerometer and Gyroscope Data with Significant Peaks")
plt.xlabel("Time")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the smoothed magnitude data with detected significant peaks
plt.figure(figsize=(15, 7))
plt.plot(timestamps, combined_magnitude_smooth, label='Combined Magnitude (Smoothed)')
plt.plot(timestamps.iloc[filtered_peaks], combined_magnitude_smooth.iloc[filtered_peaks], 'ro', label='Significant Peaks')
plt.title("Smoothed Combined Magnitude Data with Significant Peaks")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the significant peak events in a text file (time only, no date)
peak_times = [timestamps.iloc[peak].strftime("%H:%M:%S.%f")[:-3] for peak in filtered_peaks]  # Keep time, exclude date
output_txt_path = 'peak_times.txt'
with open(output_txt_path, 'w') as f:
    for time in peak_times:
        f.write(time + '\n')

print(f"Significant peak times (time only) have been saved to {output_txt_path}")
