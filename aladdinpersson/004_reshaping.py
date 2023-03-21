import torch

x = torch.arange(9)
x_view = x.view(3,3) #contiguous tensor, more efficient
y = x_view.t()  #tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])
try:
    print(1, y.view(9))
except(RuntimeError):
    print(2, y.contiguous().view(9))

x_resh = x.reshape(3,3) #doesn't matter, always works
y = x_resh.t()
# tensor([[0, 3, 6],
#        [1, 4, 7],
#        [2, 5, 8]])

try:
    print(1, y.view(9))
except(RuntimeError):
    print(2, y.contiguous().view(9))

###
print('====')
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(x1)
print(x2)

print(torch.cat((x1,x2), dim=0))
print(torch.cat((x1,x2), dim=1))

z = x1.view(-1)
print(z)

# more complicated

batch = 3
x = torch.rand((batch, 2, 5, 5))
z = x.view(batch, -1)
print(z)
print(z.shape)

z = x.permute(0, 2, 1, 3)
print(z)
print(z.shape)


x = torch.arange(10)  # shape = [10]
print(x.unsqueeze(0)) # shape = [1,10]
print(x.unsqueeze(1)) # shape = [10,1]

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # shape = [1,1,10]
print(x)
z = x.squeeze(1) # shape = [1, 10]
print(z)