# 0. Import
# 1. Create Fully Connected Network
# 2. Set device
# 3. Hyperparameters
# 4. Load Data
# 5. Initialize network
# 6. Loss and optimizer
# 7. Train Network
# 8. Check Accuracy

# Import

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Create Fully Connected Network

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Rough test
'''MINST: 28x28 = 784 image size, 10 classes'''
model = NN(784, 10)
x = torch.rand(64, 784)
print(model(x).shape)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoch = 10


# Load Data
train_set = datasets.MNIST( root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = datasets.MNIST( root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network
for epoch in range(num_epoch):
    print(f'Epoch = {epoch}')
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        '''Reshape'''
        data = data.reshape(data.shape[0], -1)

        '''Forward'''
        score = model(data)
        loss = criterion(score, targets)

        '''Backward'''
        optimizer.zero_grad()
        loss.backward()

        '''Gradient Descent'''
        optimizer.step()

# Check Accuracy
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on train")
    else:
        print("Checking accuracy on test")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += ( predictions == y ).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)