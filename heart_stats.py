import pandas as pd

# Load the new file containing start and finish times
df_final_path = 'df_final.csv'
df_final = pd.read_csv(df_final_path)

# Extracting start and finish times for the specified rounds in specific games
games_rounds = {
    "Game 1": (24, 'Paula'),
    "Game 2": (11, 'Luna'),
    "Game 3": (39, 'Luna'),
    "Game 4": (88, 'Paula'),
    "Game 5": (39, 'Luna'),
    "Game 6": (122, 'Luna')
}

games_info = {}
base_path = "RAW_DATA/RAW_game_{}_heartrate_{}.csv"

for game, (round_num, name) in games_rounds.items():
    game_number = int(game.split()[-1])
    game_data = df_final[(df_final['game'] == game_number) & (df_final['round'] == round_num)]
    start_time = df_final[df_final['game'] == game_number]['start heartrate in video_seconds'].values[0]
    finish_time = game_data['hit_seconds'].values[-1]
    file_path = base_path.format(game_number, name)
    games_info[game] = {
        "file_path": file_path,
        "time_range": (start_time, finish_time)
    }


for game, info in games_info.items():
    print(game, info)


# Function to convert 'Time Offset' to seconds
def time_offset_to_seconds(time_offset):
    h, m, s = map(float, time_offset.split(':'))
    return h * 3600 + m * 60 + s

# Function to gather heart rate data into one dataframe
def gather_heartrate_data(games_info):
    all_data = pd.DataFrame()
    
    for game, info in games_info.items():
        file_path = info["file_path"]
        start_time, end_time = info["time_range"]
        
        # Load the data, specifying the correct header row
        data = pd.read_csv(file_path, header=7)
        
        # Convert 'Time Offset' to seconds
        data['Time (s)'] = data['Time Offset'].apply(time_offset_to_seconds)
        
        # Filter the data based on the specified time range
        filtered_data = data[(data['Time (s)'] >= start_time) & (data['Time (s)'] <= end_time)]
        
        # Append the heart rate data to the combined dataframe
        all_data = all_data._append(filtered_data[['Time (s)', 'HeartRate']])
    
    return all_data


# Gather all heart rate data into one dataframe
all_heartrate_data = gather_heartrate_data(games_info)

# Calculate statistics for the combined heart rate data
mean_value = all_heartrate_data['HeartRate'].mean()
std_value = all_heartrate_data['HeartRate'].std()
min_value = all_heartrate_data['HeartRate'].min()
max_value = all_heartrate_data['HeartRate'].max()

# Save the results to a text file
with open('combined_heartrate_statistics.txt', 'w') as file:
    file.write(f"Combined Statistics for Heart Rate:\n")
    file.write(f"Mean: {mean_value}\n")
    file.write(f"Standard Deviation: {std_value}\n")
    file.write(f"Minimum: {min_value}\n")
    file.write(f"Maximum: {max_value}\n")

print("Combined statistics have been saved to 'combined_heartrate_statistics.txt'.")
