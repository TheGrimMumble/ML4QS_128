import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pandas as pd
import torchmetrics


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

def create_sequences(df, window_size, game_number):
    sequences = []
    for game, group in df.groupby('game'):
        if game in game_number:
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

# Create the sequences, Game 1 and 5 as test
window_size = 3
train_sequences = create_sequences(data, window_size, (2, 3, 4, 6))
test_sequences = create_sequences(data, window_size, (1, 5))

class GameRoundsDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, target = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(target, dtype=torch.long)

# Create dataset
train_dataset = GameRoundsDataset(train_sequences)
train_loader = DataLoader(train_dataset, batch_size=10)

test_dataset = GameRoundsDataset(test_sequences)
test_loader = DataLoader(test_dataset, batch_size=10)

"""
# Testing the input data
print(data.shape)

print(data[40:43])
print(len(train_sequences))
print(train_sequences[0])

print(data[0:3])
print(len(test_sequences))
print(test_sequences[0])

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
num_epochs = 1
batch_size = 10
learning_rate = 0.001

# Initialize metrics
accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='macro')
recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro')
f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')

"""
# Move metrics to the appropriate device
accuracy = accuracy.to(device)
precision = precision.to(device)
recall = recall.to(device)
f1 = f1.to(device)
"""


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Shape of input = batch_size, sequence_length, input_size
        
        # Define the output layer
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(x.device)
        
        # Propagate input through LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.linear(out[:, -1, :])
        return out
    


model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# Training
for epoch in range(num_epochs):  # Loop over the dataset multiple times
    for i, (features, labels) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()  # Zero the parameter gradients

        # print(f'Features: {features.size()}')
        # print(f'Labels: {labels.size()}')
        
        # Forward pass
        outputs = model(features)
        # print(f'Outputs: {outputs.size()}')
        # print(outputs)
        _, preds = torch.max(outputs, dim=1)
    
        # Update metrics
        accuracy.update(preds, labels)
        precision.update(preds, labels)
        recall.update(preds, labels)
        f1.update(preds, labels)

        loss = loss_function(outputs, labels)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()

        print(f'\n\nEpoch [{epoch+1}/{num_epochs}]\tStep [{i+1}/{len(train_loader)}]\n')

        # Compute metrics
        acc = accuracy.compute().item()
        prec = precision.compute().item()
        rec = recall.compute().item()
        f1_score = f1.compute().item()

        # Create a DataFrame to store metrics
        metrics_df = pd.DataFrame({
            'Step': [f'{i+1}/{len(train_loader)}'],
            'Loss-Value': [f'{loss.item():.3f}'],
            'Accuracy': [f'{acc:.3f}'],
            'Precision': [f'{prec:.3f}'],
            'Recall': [f'{rec:.3f}'],
            'F1 Score': [f'{f1_score:.3f}']
        })
        print(metrics_df.to_string(index=False), '\n')

        # Reset metrics after computation for the next epoch
        accuracy.reset()
        precision.reset()
        recall.reset()
        f1.reset()

        if i == 1:
            break


accuracy.reset()
precision.reset()
recall.reset()
f1.reset()

model.eval()  # Set the model to evaluation mode.
with torch.no_grad():
    for i, (features, labels) in enumerate(test_loader):
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        #_, preds = torch.max(outputs, dim=1)
    
        # Update metrics
        accuracy.update(predicted, labels)
        precision.update(predicted, labels)
        recall.update(predicted, labels)
        f1.update(predicted, labels)

    # Evaluate predictions
    metrics_df = pd.DataFrame({
        'Accuracy': [f'{acc:.3f}'],
        'Precision': [f'{prec:.3f}'],
        'Recall': [f'{rec:.3f}'],
        'F1 Score': [f'{f1_score:.3f}']
    })
    print('\n\nEvaluation on Test set\n')
    print(metrics_df.to_string(index=False), '\n')


accuracy.reset()
precision.reset()
recall.reset()
f1.reset()
