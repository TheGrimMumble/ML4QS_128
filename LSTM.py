import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import pandas as pd
import torchmetrics

# Unchangable
input_size = 31
num_classes = 5

# Hyperparameters
hidden_size = 500
num_layers = 32
num_epochs = 1
batch_size = 5
learning_rate = 0.005
window_size = 3 # number of rounds that is considered, i.e. sequence length
dropout_prob = 0.5


# Load the dataset
read_data = pd.read_csv('df_final.csv')

class_counts = read_data['target'].value_counts()
number_of_classes = len(class_counts)
total_samples = len(read_data)

# Calculate the weight for each class
weights = total_samples / (number_of_classes * class_counts)
weights_tensor = torch.tensor(weights.sort_index().values, dtype=torch.float)  # Ensure the order matches class labels



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

# Create the sequences, Game 1 and 5 as test and validation
train_sequences = create_sequences(data, window_size, (2, 3, 4, 6))
test_sequences = create_sequences(data, window_size, (0, 1))
validation_sequences = create_sequences(data, window_size, (0, 5))

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
train_loader = DataLoader(train_dataset, batch_size=batch_size)

test_dataset = GameRoundsDataset(test_sequences)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

val_dataset = GameRoundsDataset(validation_sequences)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

def testing_input_data(loader):
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



# Initialize metrics
accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='macro')
recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro')
f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')


def move_metrics_to_device(device):
    # Move metrics to the appropriate device
    accuracy = accuracy.to(device)
    precision = precision.to(device)
    recall = recall.to(device)
    f1 = f1.to(device)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0.0)
        # Apply dropout if more than one layer
        # Shape of input = batch_size, sequence_length, input_size

        # Dropout layer for applying after LSTM layers and before the linear layer
        self.dropout = nn.Dropout(dropout_prob)

        # Define the output layer
        self.linear = nn.Linear(hidden_size, num_classes)

        # Apply Xavier initialization to all weights in LSTM and Linear layers
        self.init_weights()

    def init_weights(self):
        # Initialize weights for LSTM layers
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden-hidden weights
                torch.nn.init.xavier_uniform_(param.data)
            if 'bias' in name:  # Bias initialization to zero
                param.data.fill_(0)

        # Initialize weights for the linear layer
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)  # Bias initialization to zero

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(x.device)
        
        # Propagate input through LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Apply dropout to the outputs of the LSTM layer
        out = self.dropout(out[:, -1, :])  # Apply dropout to the last time step outputs before the linear layer

        # Decode the hidden state of the last time step
        out = self.linear(out)
        return out
    


model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout_prob)

loss_function = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Add path to save the model
model_path = 'LSTM_best_model_weights.pth'

best_val_loss = float('inf')

# Create a DataFrame to store metrics
metrics_df = pd.DataFrame(columns=['Epoch', 'Step', 'Loss-Value', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Training
for epoch in range(num_epochs):  # Loop over the dataset multiple times
    for i, (features, labels) in enumerate(train_loader):
        model.train()
        # Zero the parameter gradients and forward pass
        optimizer.zero_grad()
        outputs = model(features)
        # Update metrics
        _, preds = torch.max(outputs, dim=1)
        accuracy.update(preds, labels)
        precision.update(preds, labels)
        recall.update(preds, labels)
        f1.update(preds, labels)
        # Backward and optimize
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        # Compute metrics
        acc = accuracy.compute().item()
        prec = precision.compute().item()
        rec = recall.compute().item()
        f1_score = f1.compute().item()
        # Create a DataFrame to store metrics
        batch_metrics = {
            'Epoch': f'{epoch+1}/{num_epochs}',
            'Step': f'{i+1}/{len(train_loader)}',
            'Loss-Value': f'{loss.item():.3f}',
            'Accuracy': f'{acc:.3f}',
            'Precision': f'{prec:.3f}',
            'Recall': f'{rec:.3f}',
            'F1 Score': f'{f1_score:.3f}'}
        metrics_df = metrics_df._append(batch_metrics, ignore_index=True)
        # Print only the latest row's data without the header
        if i == 0 and epoch == 0:
            # Print headers only once at the beginning
            print(' '.join([f'{col:12s}' for col in metrics_df.columns]))
        print(' '.join([f'{str(item):12s}' for item in batch_metrics.values()]))
        # Reset metrics after computation for the next epoch
        accuracy.reset()
        precision.reset()
        recall.reset()
        f1.reset()
        # Validation phase
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                val_loss = loss_function(outputs, labels)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        # Save the model if the validation loss is the best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f'Saved better model on epoch: {epoch+1}, step: {i+1} with validation loss: {avg_val_loss:.4f}')


# Initialize metrics
accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='none')
precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='none')
recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='none')
f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='none')

target_labels = {0: 'hit_correct', 1: 'hit_incorrect', 2: 'no_hit', 3: 'other_hit_correct', 4: 'other_hit_incorrect'}

model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode.
with torch.no_grad():
    for i, (features, labels) in enumerate(test_loader):
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        # Update metrics
        accuracy.update(predicted, labels)
        precision.update(predicted, labels)
        recall.update(predicted, labels)
        f1.update(predicted, labels)
    # Compute metrics
    acc = accuracy.compute()
    prec = precision.compute()
    rec = recall.compute()
    f1_score = f1.compute()
    test_df = pd.DataFrame(columns=['Class', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    for num in range(num_classes):
        # Evaluate predictions
        test_df = test_df._append({
            'Class': [target_labels[num]],
            'Accuracy': [f'{acc[num]:.3f}'],
            'Precision': [f'{prec[num]:.3f}'],
            'Recall': [f'{rec[num]:.3f}'],
            'F1 Score': [f'{f1_score[num]:.3f}']
        }, ignore_index=True)
    print('\n\nEvaluation on Test set\n')
    print(test_df.to_string(index=False), '\n')


accuracy.reset()
precision.reset()
recall.reset()
f1.reset()


