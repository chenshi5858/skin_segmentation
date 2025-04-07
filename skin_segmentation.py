import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

class SkinDataset(Dataset):
    def __init__(self, file_path):
        data_array = np.load(file_path)
        self.X = data_array[:, :3] / 255.0  
        self.y = data_array[:, 3].astype(int) 
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)  
        return sample, label

dataset = SkinDataset("skin_nskin.npy")
data_array = np.load("skin_nskin.npy")

class_weights = [0, 0]
for label in dataset.y:
    class_weights[int(label)] += 1
total_samples = len(dataset)
class_weights = [total_samples / weight for weight in class_weights]

weights = [class_weights[int(label)] for label in dataset.y]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_size = int(0.8* len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

batch_size = 32  
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

class SimpleModel(nn.Module):
    def __init__(self, input_size, n_classes=2): 
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

device = "cpu"
model = SimpleModel(3, 2).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epochs = 10
train_losses = []
val_losses = []

all_labels = []
all_preds = []

for i, (inputs, labels) in enumerate(train_loader):
    print(f"Batch {i}: Inputs size: {inputs.size()}, Labels size: {labels.size()}")
    break  

model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  
        loss = loss_fn(outputs, labels)  
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    val_loss = 0.0
    acc = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += loss_fn(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1)  
            acc += (preds == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_losses.append(val_loss / len(val_loader))
    accuracy = acc / len(val_set)

    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {accuracy:.4f}")


torch.save(model.state_dict(), "skin_segmentation.pth")

fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.show()
