import torch
import torch.nn as nn
import torch.optim as optim


# Hyperparameters
input_size = 31
sequence_length = 3 # number of rounds that is considered
hidden_size = 50
num_classes = 5
num_layers = 2
num_epochs = 2
batch_size = 100
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
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Example tensor for input and targets
inputs = torch.randn(10, 5, 10)  # (batch_size, sequence_length, input_size)
targets = torch.randn(10, 1)     # (batch_size, output_size)


for epoch in range(100):  # Loop over the dataset multiple times
    model.train()
    optimizer.zero_grad()  # Zero the parameter gradients
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')



model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predictions = model(inputs)
    # Evaluate predictions

