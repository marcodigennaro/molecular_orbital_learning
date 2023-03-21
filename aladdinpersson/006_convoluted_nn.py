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

# Create Convoluted Neural Network
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


# Rough test
'''MINST: 28x28 = 784 image size, 10 classes'''
model = CNN()
x = torch.rand(64,  1, 28, 28)
print(model(x).shape)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoch = 1


# Load Data
train_set = datasets.MNIST( root='datasets/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_set = datasets.MNIST( root='datasets/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)
model = CNN(in_channels=1).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network
for epoch in range(num_epoch):
    print(f'Epoch = {epoch}')
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        #'''Reshape'''
        #data = data.reshape(data.shape[0], -1)

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

            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += ( predictions == y ).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)