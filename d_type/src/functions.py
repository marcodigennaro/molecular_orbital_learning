import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for count, (inputs, targets) in enumerate(data_loader):
        print(f'Count/batch_size = {count}/{data_loader.batch_size}')
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.float()
        #targets = targets.float()

        print('inputs.shape:')
        print(inputs.shape)

        print('inputs[0][0]:')
        print(inputs[0][0])
        print("Type of inputs:", type(inputs))

        print('targets.shape:')
        print(targets.shape)

        print('targets:')
        print(targets)

        # calculate loss
        predictions = model(inputs)
        print('predictions.shape:')
        print(predictions.shape)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss. item()}")

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print("------------------")
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
    print(f"Train completed")

def liear_model(X_train, input_size, hidden_size, output_size):

    w1 = torch.rand(input_size, hidden_size, requires_grad=True)
    w2 = torch.rand(hidden_size, output_size, requires_grad=True)

    y_pred = X_train.mm(w1).mm(w2)
    loss = (y_pred - Y_train).pow(2).sum()

    #if iter % 10 == 0:
    print(iter, loss.item(), w1 , w2)
    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()

    return w1, w2

def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    validation_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, validation_data