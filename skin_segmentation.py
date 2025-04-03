import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.utils.data as data
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class SimpleModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super(SimpleModel, self).__init__()
        self.input_layer = nn.Linear(input_size, 8)
        self.hidden_layer_1 = nn.Linear(8, 16)
        self.output_layer = nn.Linear(16, n_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.relu(self.hidden_layer_1(x))
        x = self.sigmoid(self.output_layer(x))
        return x


epochs = 10
batch_size = 16
train_split = 0.7
device = 'cpu'

dataset = np.load("skin_nskin.npy")
n_train = int(len(dataset) * train_split)
n_val = len(dataset) - n_train
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

model = SimpleModel(3, 1)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

losses = []
val_losses = []
epoch_accuracy = []

model.to(device)
model.train()

for epoch in range(epochs):
  acc = 0.0
  val_loss = 0.0
  epoch_loss = 0.0

  for i, data in enumerate(train_loader):
      inputs, labels = data

      inputs = inputs.to(device=device, dtype=torch.float32)
      labels = labels.to(device=device, dtype=torch.long)

      optimizer.zero_grad()

      outputs = model(inputs)

      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()/len(train_loader)

  losses.append(epoch_loss)

  with torch.no_grad():
    for i, data in enumerate(val_loader):
      inputs, labels = data

      inputs = inputs.to(device=device, dtype=torch.float32)
      labels = labels.to(device=device, dtype=torch.long)

      outputs = model(inputs)
      val_loss += loss_fn(outputs, labels).item()/len(val_loader)

      acc += (outputs.argmax(dim=1) == labels).sum().item()

    epoch_accuracy.append(acc/len(val_loader))
    val_losses.append(val_loss)



plt.plot(losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()