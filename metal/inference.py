import torch
from train import FeedForwardNet, download_mnist_datasets

class_mapping = [ str(i) for i in range(9) ]

def predict(model, input, target, class_mapping):
    model.eval()
    #model.train()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.001, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    return predicted, expected


if __name__ == "__main__":
    # load back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    # load MINST validation data
    _, validation_data = download_mnist_datasets()

    # get a sample from dataset
    input, target = validation_data[0][0], validation_data[0][1]

    # make an inference
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)

    print(f"predicted : {predicted}, expected : {expected}")