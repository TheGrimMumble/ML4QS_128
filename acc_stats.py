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
base_path = "RAW_DATA/RAW_game_{}_acceleration_{}.csv"

for game, (round_num, name) in games_rounds.items():
    game_number = int(game.split()[-1])
    game_data = df_final[(df_final['game'] == game_number) & (df_final['round'] == round_num)]
    start_time = df_final[df_final['game'] == game_number]['accelarator first signal_seconds'].values[0]
    finish_time = game_data['hit_seconds'].values[-1]
    file_path = base_path.format(game_number, name)
    games_info[game] = {
        "file_path": file_path,
        "time_range": (start_time, finish_time)
    }


for game, info in games_info.items():
    print(game, info)


# Function to analyze the acceleration and gather data into one dataframe
def gather_acceleration_data(games_info):
    all_data = pd.DataFrame()
    
    for game, info in games_info.items():
        file_path = info["file_path"]
        start_time, end_time = info["time_range"]
        
        # Load the data
        data = pd.read_csv(file_path)
        
        # Filter the data based on the specified time range
        filtered_data = data[(data['Time (s)'] >= start_time) & (data['Time (s)'] <= end_time)]
        
        # Append the absolute acceleration data to the combined dataframe
        all_data = all_data._append(filtered_data[['Time (s)', 'Absolute acceleration (m/s^2)']])
    
    return all_data

# Gather all absolute acceleration data into one dataframe
all_acceleration_data = gather_acceleration_data(games_info)

# Calculate statistics for the combined absolute acceleration data
mean_value = all_acceleration_data['Absolute acceleration (m/s^2)'].mean()
std_value = all_acceleration_data['Absolute acceleration (m/s^2)'].std()
min_value = all_acceleration_data['Absolute acceleration (m/s^2)'].min()
max_value = all_acceleration_data['Absolute acceleration (m/s^2)'].max()

# Save the results to a text file
with open('combined_acceleration_statistics.txt', 'w') as file:
    file.write(f"Combined Statistics for Absolute Acceleration:\n")
    file.write(f"Mean: {mean_value}\n")
    file.write(f"Standard Deviation: {std_value}\n")
    file.write(f"Minimum: {min_value}\n")
    file.write(f"Maximum: {max_value}\n")

print("Combined statistics have been saved to 'combined_acceleration_statistics.txt'.")
