import torch
from torch import nn, optim
from Project2.load_data_tester import dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    # refrence: https://github.com/kohyar/LTTng_LSTM_Anomaly_Detection
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)

        lstm_out, _ = self.lstm(x, (h0, c0))

        out = self.fc(lstm_out[:, -1, :])

        return out


def train_model(model, train_loader, epochs, optimizer, loss_function):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        outputs = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model.forward(x)
            output = output.squeeze(1)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            if i % (batch_size*10) == 0:
                print(
                    f'Epoch {epoch} Batch {i//batch_size} loss: {loss.item()}')
            epoch_losses.append(loss.item())
            outputs.append(output)
        losses.append(epoch_losses)
    return losses, outputs


input_size = len(dataset.data[0][0])
print(input_size)
hidden_size = 100
num_layers = 1
output_size = 1
learning_rate = 0.001
batch_size = 32
epochs = 10

model = LSTM(input_size, hidden_size, num_layers, output_size)
loss_function = nn.MSELoss()
adam = optim.Adam(model.parameters(), learning_rate)
train_dataloader = DataLoader(dataset, batch_size, shuffle=False)

losses, output = train_model(
    model, train_dataloader, epochs, adam, loss_function)

print(output[-1])
import matplotlib.pyplot as plt

plt.plot([sum(epoch_losses)/len(epoch_losses) for epoch_losses in losses])
# new_epochs = range(0, len(losses), len(losses)//epochs)
# for epoch in new_epochs:
#     plt.axvline(x=epoch, color='r', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

plt.plot(losses[-1])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss in Last Epoch')
plt.show()

