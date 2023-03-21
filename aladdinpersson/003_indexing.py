import torch

batch_size = 10
features = 25

x = torch.rand((batch_size, features))

print(x[0].shape)

x[0, 0] = 100

x = torch.arange(10)
indices = [2, 3, 6]
print(x[indices])

x = torch.rand((3,5))
print(x)
rows = torch.tensor([1,0])
cols = torch.tensor([0,1])
print(x[rows, cols])


# More advanced indexing
x = torch.arange(10)
print(x[(x<2) | (x>8)])
print(x[x.remainder(2)==0])

# Useful operation
print(torch.where(x>5, x, x*2))
print(torch.tensor([0, 0, 21,2,4,65,6,2,1,2]).unique())

print(x.ndimension())
print(x.numel())
