import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the stacked LSTM model class
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x, (hidden, cell) = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
    
# Define a function to train the model
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    train_losses = []
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
    return train_losses

# Define a function to evaluate the model
def evaluate(model, dataloader, criterion, device):
    model.eval()
    eval_losses = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            eval_losses.append(loss.item())
            
    return eval_losses

# Define a function to plot the loss curve
def plot_loss_curve(train_losses, eval_losses, title='Loss Curve'):
    plt.plot(train_losses, label='Train')
    plt.plot(eval_losses, label='Eval')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Initialize the model, optimizer, criterion and move the model to GPU
input_size = ...
hidden_size = ...
num_layers = ...
output_size = ...
dropout = ...

model = StackedLSTM(input_size, hidden_size, num_layers, output_size, dropout)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = ...
criterion = ...