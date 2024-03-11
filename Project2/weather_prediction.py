import torch
from torch import nn, optim
from load_data import train_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    # refrence: https://github.com/kohyar/LTTng_LSTM_Anomaly_Detection
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        
        return out

model = LSTM(input_size=1, hidden_size=100, num_layers=1, output_size=1).to(device)
print(model)

loss_function = nn.CrossEntropyLoss()
adam = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, epochs, optimizer, loss_function):
    model.train()
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch} Iteration {i} loss: {loss.item()}')
                
train_model(model, train_dataloader, 10, adam, loss_function)