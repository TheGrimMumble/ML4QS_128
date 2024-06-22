import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import torchmetrics
from sklearn.preprocessing import StandardScaler
import numpy as np


# Input Output
input_size = 8
num_classes = 5
load_dataset = pd.read_csv('df_final.csv')
game = 'game'
round = 'round'
target = 'target'
selected_features = ['game', 'round', # game and round must be first for dataprocessing
    'max_acc_value', 'games played prior on current day', 'experience score', 'after2_mean', 'after3_min',
    'before1_mean', 'after3_max', 'winner_streak',
    'target'] # target must be the last item
dataset = load_dataset[selected_features]
target_labels = {0: 'hit_correct',
                 1: 'hit_incorrect',
                 2: 'no_hit',
                 3: 'other_hit_correct',
                 4: 'other_hit_incorrect'}

# CHOICES
# Train, Validation, Test
random_data_split = False
if random_data_split:
    train_split = 0.7
    validation_split = 0.15
    test_split = 0.15
else: # split based on game number, ignore 0
    train_split = (0, 2, 3, 4, 6)
    validation_split = (0, 5)
    test_split = (0, 1)
# Standarization of inputs
standarization = False
# Shuffle dataloaders
shuffle_train = False
shuffle_validation = False
shuffle_test = False
# Model choices
use_attention = False
use_dropout = False
use_xavier_init = False

# Hyperparameters
num_epochs = 64
batch_size = 32
window_size = 2 # sequence length (i.e. number of rounds)
hidden_size = 64
num_layers = 16
learning_rate = 0.005
dropout_prob = 0.5

# Loss function
alpha_loss = 0 # loss function's weight scaling, 1: proportional, 0: uniform
focal_loss = False
if focal_loss:
    gamma_loss = 3.5 # higher values penalizes easy classes

# Miscellaneous
num_workers = 0 # multiprocessing, 0: off
eval_on_val_set = True
model_path = 'LSTM_best_model_weights.pth' # for saving weights


def get_weights_tensor(dataset, target_var_name, alpha_loss):
    class_counts = dataset[target_var_name].value_counts()
    number_of_classes = len(class_counts)
    total_samples = len(dataset)
    # Calculate the proportional weight for each class
    proportional_weights = total_samples / (number_of_classes * class_counts)
    # Convert proportional_weights to a tensor
    proportional_weights_tensor = torch.tensor(proportional_weights.sort_index().values, dtype=torch.float)
    # Generate uniform weights using ones_like on the tensor
    uniform_weights_tensor = torch.ones_like(proportional_weights_tensor)
    # Adjust weights based on alpha
    weights_tensor = alpha_loss * proportional_weights_tensor + (1 - alpha_loss) * uniform_weights_tensor
    return weights_tensor


def sequences_random_split(df, window_size, game, round, target, train_split, validation_split, test_split):
    if train_split + validation_split + test_split != 1:
        return print("Train, Validation and Test split incorrect.")
    sequences = []
    for _game_, group in df.groupby(game):
        if _game_ in df.groupby(game).groups.keys():
            # Ensure the data is sorted by round within each game
            group = group.sort_values(round)
            # Drop the game and round columns for modeling
            features = group.drop(columns=[game, round])
            num_sequences = len(group) // window_size
            for i in range(num_sequences):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                sequence = features.iloc[start_idx:end_idx]
                if len(sequence) == window_size:
                    _target_ = sequence.iloc[-1][target]
                    sequence = sequence.drop(columns=target)
                    sequences.append((sequence.values, _target_))
    # Shuffle the sequences
    np.random.shuffle(sequences)
    # Split into train, validation, and test sets
    train_size = int(len(sequences) * train_split)
    validation_size = int(len(sequences) * validation_split)
    train_sequences = sequences[:train_size]
    validation_sequences = sequences[train_size:train_size + validation_size]
    test_sequences = sequences[train_size + validation_size:]
    return train_sequences, validation_sequences, test_sequences


def rolling_window_sequences(df, window_size, game, round, target, game_numbers):
    sequences = []
    for _game_, group in df.groupby(game):
        if _game_ in game_numbers:
            # Ensure the data is sorted by round within each game
            group = group.sort_values(round)
            # Drop the game and round columns for modeling
            features = group.drop(columns=[game, round])
            # Create sequences
            for i in range(0, len(group) - window_size + 1):
                sequence = features.iloc[i:i+window_size]
                # Use the target of the last round in the window
                _target_ = sequence.iloc[-1][target]
                # Remove the target from features
                sequence = sequence.drop(columns=target)
                sequences.append((sequence.values, _target_))
    return sequences


def standardize_sequences(sequences):
    # Unpack the sequences into features and targets
    features, targets = zip(*sequences)
    # Combine all features into a single array for fitting the scaler
    combined_features = np.vstack(features)
    # Initialize and fit the scaler
    scaler = StandardScaler()
    scaler.fit(combined_features)
    # Transform the features
    standardized_features = [scaler.transform(f) for f in features]
    # Recombine the standardized features with their targets
    standardized_sequences = list(zip(standardized_features, targets))
    cached_standardized_seq = standardized_sequences.copy()
    return cached_standardized_seq


class GameRoundsDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence, target = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(target, dtype=torch.long)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lstm_output, hidden):
        # Linear transformation
        linear_out = self.linear_in(lstm_output)
        # Dot product between the hidden state and each time step of the LSTM outputs
        scores = torch.bmm(linear_out, hidden.unsqueeze(2)).squeeze(2)
        # Apply softmax to get attention weights
        attention_weights = self.softmax(scores)
        # Multiply the attention weights with the lstm output to get the context vector
        context_vector = torch.bmm(lstm_output.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return context_vector, attention_weights


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 dropout_prob, use_attention, use_dropout, use_xavier_init):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_dropout = use_dropout
        self.use_xavier_init = use_xavier_init
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout_prob if num_layers > 1 else 0.0)
        if use_attention:
            # Attention layer
            self.attention = Attention(hidden_size)
        if use_dropout:
            # Dropout layer for applying after LSTM layers and before the linear layer
            self.dropout = nn.Dropout(dropout_prob)
        # Define the output layer
        self.linear = nn.Linear(hidden_size, num_classes)
        if use_xavier_init:
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
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))  # lstm_out: tensor of shape (batch_size, seq_length, hidden_size)
        if self.use_attention:
            # Last hidden state
            hn = hn[-1]  # Considering only the last layer's hidden state
            # Attention layer
            context_vector, attention_weights = self.attention(lstm_out, hn)
        else:
            context_vector = lstm_out[:, -1, :]  # Use the last time step's output for classification
        if self.use_dropout:
            # Apply dropout to the context vector
            context_vector = self.dropout(context_vector)
        # Decode the hidden state of the last time step
        out = self.linear(context_vector)
        return out  #, attention_weights # these weights are only used for interpretability if needed


class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can be a float or a tensor with class weights

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def training_model(model, optimizer, loss_function, num_epochs, train_loader, validation_loader):
    # Initialize metrics for training and evaluation
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='macro')
    recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='macro')
    f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
    best_val_loss = float('inf')
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
            if eval_on_val_set == True:
                # Validation phase
                model.eval()  # Set model to evaluation mode
                total_val_loss = 0
                with torch.no_grad():
                    for features, labels in validation_loader:
                        outputs = model(features)
                        val_loss = loss_function(outputs, labels)
                        total_val_loss += val_loss.item()
                avg_val_loss = total_val_loss / len(validation_loader)
                # Save the model if the validation loss is the best
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), model_path)
                    print(f'Saved better model on epoch: {epoch+1}, step: {i+1} with validation loss: {avg_val_loss:.4f}')
            else:
                torch.save(model.state_dict(), model_path)
    return best_val_loss


def evaluate_model(model, test_loader):
    # Initialize metrics for testing
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='none')
    precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='none')
    recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='none')
    f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='none')
    # Set up model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode.
    with torch.no_grad():
        for i, (features, labels) in enumerate(test_loader):
            outputs = model(features)
            _, predicted = torch.max(outputs, dim=1)
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


def run_manual():
    weights_tensor = get_weights_tensor(dataset, target, alpha_loss)

    if random_data_split:
        train_sequences, validation_sequences, test_sequences = sequences_random_split(dataset, window_size, game, round, target, train_split, validation_split, test_split)
    else:
        train_sequences = rolling_window_sequences(dataset, window_size, game, round, target, train_split)
        validation_sequences = rolling_window_sequences(dataset, window_size, game, round, target, validation_split)
        test_sequences = rolling_window_sequences(dataset, window_size, game, round, target, test_split)

    if standarization:
        train_sequences = standardize_sequences(train_sequences)
        validation_sequences = standardize_sequences(validation_sequences)
        test_sequences = standardize_sequences(test_sequences)

    # Create datasets
    train_dataset = GameRoundsDataset(train_sequences)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)

    validation_dataset = GameRoundsDataset(validation_sequences)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle_validation, num_workers=num_workers)

    test_dataset = GameRoundsDataset(test_sequences)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

    # Instantiate the model
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, 
                    dropout_prob, use_attention, use_dropout, use_xavier_init)
    if focal_loss:
        loss_function = FocalLoss(gamma_loss, alpha=weights_tensor)
    else:
        loss_function = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    val_loss = training_model(model, optimizer, loss_function, num_epochs, train_loader, validation_loader)

    # Evaluate model
    evaluate_model(model, test_loader)


# run_manual()


import optuna
import joblib


def objective(trial):
    # Hyperparameters to be optimized
    random_data_split = trial.suggest_categorical("random_data_split", [True, False])
    train_split = trial.suggest_categorical("train_split", [0.5, 0.7])
    standarization = trial.suggest_categorical("standarization", [True, False])
    shuffle_train = trial.suggest_categorical("shuffle_train", [True, False])
    shuffle_validation = trial.suggest_categorical("shuffle_validation", [True, False])
    shuffle_test = trial.suggest_categorical("shuffle_test", [True, False])
    use_attention = trial.suggest_categorical("use_attention", [True, False])
    use_dropout = trial.suggest_categorical("use_dropout", [True, False])
    use_xavier_init = trial.suggest_categorical("use_xavier_init", [True, False])
    num_epochs = trial.suggest_categorical("num_epochs", [48, 80])
    batch_size = trial.suggest_categorical("batch_size", [24, 32])
    window_size = trial.suggest_categorical("window_size", [2, 4])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers = trial.suggest_categorical("num_layers", [8, 16, 32])
    learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.005, 0.01])
    dropout_prob = trial.suggest_categorical("dropout_prob", [0.3, 0.4, 0.5])
    alpha_loss = trial.suggest_categorical("alpha_loss", [0, 0.4, 0.7, 1])
    focal_loss = trial.suggest_categorical("focal_loss", [True, False])
    gamma_loss = trial.suggest_categorical("gamma_loss", [2.0, 2.5, 3.0, 3.5])

    # Load dataset, define splits based on random_data_split
    # Create the data splits based on `random_data_split` and `train_split`
    if random_data_split:
        validation_split, test_split = ((1 - train_split) / 2, (1 - train_split) / 2)
        train_sequences, validation_sequences, test_sequences = sequences_random_split(
            dataset, window_size, game, round, target, train_split, validation_split, test_split)
    else:
        train_split, validation_split, test_split = (0, 2, 3, 4, 6), (0, 5), (0, 1)
        # Data preprocessing, etc.
        train_sequences = rolling_window_sequences(dataset, window_size, game, round, target, train_split)
        validation_sequences = rolling_window_sequences(dataset, window_size, game, round, target, validation_split)
        test_sequences = rolling_window_sequences(dataset, window_size, game, round, target, test_split)

    if standarization:
        train_sequences = standardize_sequences(train_sequences)
        validation_sequences = standardize_sequences(validation_sequences)
        test_sequences = standardize_sequences(test_sequences)

    # DataLoader
    train_loader = DataLoader(GameRoundsDataset(train_sequences), batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    validation_loader = DataLoader(GameRoundsDataset(validation_sequences), batch_size=batch_size, shuffle=shuffle_validation, num_workers=num_workers)
    test_loader = DataLoader(GameRoundsDataset(test_sequences), batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)

    # Model instantiation
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout_prob, use_attention, use_dropout, use_xavier_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights_tensor = get_weights_tensor(dataset, target, alpha_loss)
    
    if focal_loss:
        loss_function = FocalLoss(gamma_loss, alpha=weights_tensor)
    else:
        loss_function = nn.CrossEntropyLoss(weight=weights_tensor)

    # Here you could add your training and validation logic
    # For simplicity, let's say it returns the validation loss
    val_loss = training_model(model, optimizer, loss_function, num_epochs, train_loader, validation_loader)
    return val_loss


def save_best_trial(study, trial):
    if study.best_trial.number == trial.number:
        # Save the best model
        joblib.dump(trial.params, 'best_LSTM_params.pkl')
        print(f"Saved new best parameters with loss-value: {trial.value}")


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1, callbacks=[save_best_trial])  # Adjust n_trials for the number of combinations you want to explore

trial = study.best_trial
print(f"\nBest trial:\n{trial}")

print(f"  Value: {trial.value}")

print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
