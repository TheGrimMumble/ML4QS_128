import pandas as pd

# Load the CSV file
# file_path = '/mnt/data/df_FINAL-5.csv'
df = pd.read_csv('df_final.csv')

# Display the first few rows of the dataframe to understand its structure
# df.head()

# Filter the dataframe to include only rows where 'target' is 0, 1, 3, or 4
filtered_df = df[df['target'].isin([0, 1, 3, 4])]
print(len(filtered_df))

# Convert the 'duration round_seconds' column to numeric, errors='coerce' will convert non-numeric values to NaN
# filtered_df['duration round_seconds'] = pd.to_numeric(filtered_df['duration round_seconds'], errors='coerce')
print(filtered_df['duration round_seconds'].notna().sum())

# Calculate the mean, standard deviation, min, and max of 'duration round_seconds' for the filtered rows
mean_duration = filtered_df['duration round_seconds'].mean()
std_duration = filtered_df['duration round_seconds'].std()
min_duration = filtered_df['duration round_seconds'].min()
max_duration = filtered_df['duration round_seconds'].max()

print('mean', mean_duration)
print('std', std_duration)
print('min', min_duration)
print('max', max_duration)
