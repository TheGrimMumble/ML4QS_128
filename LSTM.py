import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pandas as pd


# Load the dataset
read_data = pd.read_csv('df_final.csv')

# Selecting the required features and target
columns = [
    'game', 'round',
    'confidence score', 'experience score', 'games played prior on current day',
    'winner_streak', 'prev_round_hit_corr', 'favorite fruit_prume', 'favorite fruit_strawberry',
    'accelarator device_band', 'accelarator device_elastics', 'duration round_seconds',
    'before3_mean', 'before3_min', 'before3_max',
    'before2_mean', 'before2_min', 'before2_max',
    'before1_mean', 'before1_min', 'before1_max',
    'after1_mean', 'after1_min', 'after1_max',
    'after2_mean', 'after2_min', 'after2_max',
    'after3_mean', 'after3_min', 'after3_max',
    'max_acc_value', 'max_acc_time', 'time_diff_max_acc',
    'target'
]

data = read_data[columns]

def create_sequences(df, window_size):
    sequences = []
    for _, group in df.groupby('game'):
        # Ensure the data is sorted by round within each game
        group = group.sort_values('round')
        # Drop the game and round columns for modeling
        features = group.drop(columns=['game', 'round'])
        # Create sequences
        for i in range(0, len(group) - window_size + 1):
            sequence = features.iloc[i:i+window_size]
            # Use the target of the last round in the window
            target = sequence.iloc[-1]['target']
            # Remove the target from features
            sequence = sequence.drop(columns='target')
            sequences.append((sequence.values, target))
    return sequences

# Create the sequences
window_size = 3
sequences = create_sequences(data, window_size)

class GameRoundsDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, target = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(target, dtype=torch.float)

# Create dataset
dataset = GameRoundsDataset(sequences)
loader = DataLoader(dataset, batch_size=10)

"""
# Testing the input data
print(data.shape)
print(data[0:3])

print(len(sequences))
print(sequences[0])

for i, (inputs, targets) in enumerate(loader):
    print(f"Batch {i+1}")
    print("Inputs Shape:", inputs.shape)
    print("Targets Shape:", targets.shape)
    print("Inputs Example:", inputs[0])
    print("Targets Example:", targets[0])
    
    if i == 0:  # Only inspect the first batch
        break
"""

# Hyperparameters
input_size = 31
sequence_length = 3 # number of rounds that is considered
hidden_size = 10
num_classes = 5
num_layers = 2
num_epochs = 2
batch_size = 10
learning_rate = 0.001
output_size = 1


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Shape of input = batch_size, sequence_length, input_size
        
        # Define the output layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Propagate input through LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.linear(out[:, -1, :])
        return out
    


model = LSTMModel(input_size, hidden_size, num_layers, num_classes, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Example tensor for input and targets
example_inputs = torch.randn(batch_size, 5, 10)  # (batch_size, sequence_length, input_size)
example_targets = torch.randn(batch_size, output_size)     # (batch_size, output_size)


for epoch in range(num_epochs):  # Loop over the dataset multiple times
    for i, (features, labels) in enumerate(loader):
        model.train()
        optimizer.zero_grad()  # Zero the parameter gradients
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}]\nStep [{i+1}/{len(loader)}] Loss: {loss.item():.4f}')



model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for i, (features, labels) in enumerate(loader):
        predictions = model(features)
        print(f"Predicted [{predictions}]\nTruth [{labels}]")
    # Evaluate predictions

