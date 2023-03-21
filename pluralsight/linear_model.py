import numpy as np
import matplotlib.pyplot as plt
import torch
import random

# No optimizer used in this tutorial

mu = 100
sigma = 50
x = np.array([ [random.gauss(mu, sigma)] for i in range(10) ], dtype=np.float32)
y = np.array([ _ + random.gauss(0, 50) for _ in x ])

X_train = torch.from_numpy(x)
Y_train = torch.from_numpy(y)

input_size = 1
hidden_size = 1
output_size = 1
learning_rate = 1e-6

w1 = torch.rand( input_size, hidden_size, requires_grad=True )
w2 = torch.rand( hidden_size, output_size, requires_grad=True )
print('w1 = ', w1)
print('w1.shape = ', w1.shape)
print('w2 = ', w2)
print('w2.shape = ', w2.shape)

for iter in range(1, 10):
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


x_train_tensor = torch.from_numpy(x)
print('x_train_tensor', x_train_tensor)

predicted_in_tensor = x_train_tensor.mm(w1).mm(w2)
print('predicted_in_tensor', predicted_in_tensor)

predicted = predicted_in_tensor.detach().numpy()
print('predicted', predicted)

plt.figure(figsize=(12, 8))
plt.scatter(x, y, label = 'Original data', s=250, c='g')
plt.plot(x, predicted, label = 'predicted')
plt.legend()
plt.show()