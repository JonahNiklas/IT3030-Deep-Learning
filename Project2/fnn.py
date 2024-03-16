import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn as nn
from config import *
from load_data import getTrainingSet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, size_of_hidden_layers, output_size):
        super(FeedForwardNN, self).__init__()
        self.size_of_hidden_layers = size_of_hidden_layers
        self.fc1 = nn.Linear(input_size, size_of_hidden_layers[0])
        for i in range(1, len(size_of_hidden_layers)):
            setattr(
                self, f'fc{i+1}', nn.Linear(size_of_hidden_layers[i-1], size_of_hidden_layers[i]))

        self.out = nn.Linear(size_of_hidden_layers[-1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        for i in range(1, len(self.size_of_hidden_layers)):
            fc= getattr(self, f'fc{i+1}')
            x = self.relu(fc(x))
        x = self.out(x)
        return x


def train_model(model, train_loader, epochs, optimizer, loss_function, verbose=False, save_model=True):
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        outputs = []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # unroll the sequence
            x = x.view(x.size(0), -1)
            output = model.forward(x)
            output = output.squeeze(1)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            
            if verbose and i % (batch_size*10) == 0:
                print(
                    f'Epoch {epoch} Batch {i//batch_size} loss: {loss.item()}')
            epoch_losses.append(loss.item())
            outputs.append(output)
        losses.append(np.mean(epoch_losses))
        print(f'Epoch {epoch} loss: {losses[-1]}')
    if save_model:
        torch.save(model.state_dict(), 'model.pth')
    return losses, outputs


dataset = getTrainingSet(reshape=False, sequence_length=24)
features = len(dataset.X[0][0])
sequence_length = len(dataset.X[0])
input_size = features * sequence_length
size_of_hidden_layers = SIZE_OF_HIDDEN_LAYERS_FNN
output_size = OUTPUT_SIZE
epochs = EPOCHS_FNN
learning_rate = LEARNING_RATE
batch_size = BATCH_SIZE

model = FeedForwardNN(input_size, size_of_hidden_layers, output_size)
loss_function = nn.MSELoss()
adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=False)
losses, outputs = train_model(
    model,
    train_dataloader,
    epochs,
    adam,
    loss_function,
    verbose=True
)

plt.plot(losses)
# new_epochs = range(0, len(losses), len(losses)//epochs)
# for epoch in new_epochs:
#     plt.axvline(x=epoch, color='r', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# plt.plot(losses[-1])
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Loss in Last Epoch')
# plt.show()


