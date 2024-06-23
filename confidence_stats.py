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
base_path = "RAW_DATA/RAW_game_{}_labels_and_variables_{}.csv"

for game, (round_num, name) in games_rounds.items():
    game_number = int(game.split()[-1])
    file_path = base_path.format(game_number, name)
    games_info[game] = {
        "file_path": file_path,
        "last_round": round_num
    }


for game, info in games_info.items():
    print(game, info)


# Function to gather confidence score data into one dataframe
def gather_confidence_data(games_info):
    all_data = pd.DataFrame()
    
    for game, info in games_info.items():
        file_path = info["file_path"]
        last_round = info["last_round"]
        
        # Load the data, specifying the correct header row
        data = pd.read_csv(file_path)
        
        # Filter the data based on the specified last round
        filtered_data = data[data['round'] <= last_round]
        
        # Append the confidence score data to the combined dataframe
        all_data = all_data._append(filtered_data[['round', 'confidence score']])
    
    return all_data


# Gather all confidence score data into one dataframe
all_confidence_data = gather_confidence_data(games_info)

# Calculate statistics for the combined confidence score data
mean_value = all_confidence_data['confidence score'].mean()
std_value = all_confidence_data['confidence score'].std()
min_value = all_confidence_data['confidence score'].min()
max_value = all_confidence_data['confidence score'].max()

# Calculate the percentage of missing values
missing_percentage = all_confidence_data['confidence score'].isna().mean() * 100

# Save the results to a text file
with open('combined_confidence_statistics.txt', 'w') as file:
    file.write(f"Combined Statistics for Confidence Score:\n")
    file.write(f"Mean: {mean_value}\n")
    file.write(f"Standard Deviation: {std_value}\n")
    file.write(f"Minimum: {min_value}\n")
    file.write(f"Maximum: {max_value}\n")
    file.write(f"Percentage of Missing Values: {missing_percentage}%\n")

print("Combined statistics have been saved to 'combined_confidence_statistics.txt'.")
