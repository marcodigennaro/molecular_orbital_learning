from torch import nn
import torch.nn.functional as F

class NN(nn.Module):
    '''
    Fully Connected Network example from:
    https://www.youtube.com/watch?v=Jy4wM2X21u0&ab_channel=AladdinPersson
    '''
    def __init__(self, num_classes=10):
        super(NN, self).__init__()
        self.input_size = image_width*image_height
        print(f'image size = {image_width}x{image_height}')
        print(f'hidden neurons = {hidden_neurons}')
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(self.input_size, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        # x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        return x

class CNN(nn.Module):
    '''
    Convoluted Neural Network from:
    https://www.youtube.com/watch?v=wnK3uWv_WkU&ab_channel=AladdinPersson
    '''

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

class mdg_FeedForwardNet(nn.Module):
    '''
    Class NN from TheSoundofAI youtube channel
    https://www.youtube.com/watch?v=4p0G6tgNLis&ab_channel=ValerioVelardo-TheSoundofAI
    '''
    def __init__(self):
        super().__init__()
        colors = 4
        image_width = 25
        image_height = 25
        num_classes = 5

        #colors = 1
        #image_width = 28
        #image_height = 28
        #num_classes = 10

        hidden_neurons = 256

        img_size = colors * image_width * image_height
        self.flatten = nn.Flatten()
        self.dense_layer = nn.Sequential(
            nn.Linear( img_size,
                       hidden_neurons),
            nn.ReLU(),
            nn.Linear( hidden_neurons, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flatten_data = self.flatten(input_data)
        print(f'flatten_data.shape: {flatten_data.shape}')
        print(flatten_data[0])
        #print(self.dense_layer.weight.dtype)

        logits = self.dense_layer(flatten_data)
        print(f'logits.shape: {logits.shape}')
        predictions = self.softmax(logits)
        print(f'predictions.shape: {predictions.shape}')
        return predictions

class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layer = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flatten_data = self.flatten(input_data)
        logits = self.dense_layer(flatten_data)
        predictions = self.softmax(logits)
        return predictions

class ImageClassifier(nn.Module):
    '''
    Image Classifier Neural Network example from:
    https://www.youtube.com/watch?v=mozBidd58VQ&ab_channel=NicholasRenotte
    '''
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10)
        )

    def forward(self, x):
        return self.model(x)
