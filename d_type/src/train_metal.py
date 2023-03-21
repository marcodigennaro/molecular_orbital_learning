import torch
from torch.utils.data import DataLoader
from class_NN import *
from functions import *
from proj_paths import FIGS_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS

# 1- download dataset
# 2- create data loader
# 3- build model
# 4- train
# 5- save trained model

if __name__ == "__main__":

    # download data and create data loader
    train_data, _ = download_mnist_datasets()

    # create data loader
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # build model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    feed_forward_net = mdg_FeedForwardNet().to(device)

    # instantiate loss_fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model trained and stored")