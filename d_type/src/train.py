import os.path
from pathlib import Path
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms as T
from torchvision.io import read_image

from class_orbitals import Orbitals
from class_NN import *
from functions import train
from proj_paths import FIGS_DIR, BATCH_SIZE, LEARNING_RATE, EPOCHS
import matplotlib.pyplot as plt

# 1- download dataset
# 2- create data loader
# 3- build model
# 4- train
# 5- save trained model



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('distorted_orbitals.csv', index_col=0)
    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    print(f'Length df = {len(df)}\n       train = {train_size}\n       test = {test_size}')

    dataset = Orbitals(annotations_file='distorted_orbitals.csv',
                       img_dir=FIGS_DIR,
                       transform=T.Compose([
                                T.Resize(size=25, antialias=True),
                                T.CenterCrop(size=25),
                                #T.ToTensor()
                                ])
                       )

    #sample_data = dataset[0]
    #features, label = sample_data

    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_set,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              #train=True
                              )

    test_loader = DataLoader(dataset=test_set,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             #train=False
                             )

    sample_features, sample_label = next(iter(train_loader))
    print(f'Feature sample: {sample_features.size()}')
    print(f'Label sample: {sample_label.size()}')
    img = sample_features[0]
    label = sample_label[0]
    img = T.ToPILImage()(img)
    image_width, image_height = T.functional.get_image_size(img)  # 480x480

    # build model
    model = mdg_FeedForwardNet().to(device)

    # instantiate loss_fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(model, train_loader, loss_fn, optimizer, device, EPOCHS)

    torch.save(model.state_dict(), "model.pth")
    print("Model trained and stored")