import torch

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# Addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)
# or
z2 = torch.add(x,y)
# or
z3 = x + y

# Division
z = torch.true_divide(x,y)

# Inplace operation
t = torch.zeros(3)
t.add_(x)
t += x

# Exponentiation
z = x.pow(2)
z = x**2
print(z)

# Comparison
z = x > 0
print(z)

# Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1, x2)
#or
x3 = x1.mm(x2)

# Matrix Exponentiantion
matrix_exp = torch.rand((3,3))
mp = matrix_exp.matrix_power(4)
print(mp)

# Elementwise multiplication
z = x * y
print(z)

# Dot Product
z = torch.dot(x,y)
print(z)

# Batch Matrix Multiplication
batch = 3
n = 2
m = 3
p = 4

#t1 = torch.rand((batch, n, m))
#t2 = torch.rand((batch, m, p))
t1 = torch.tensor([[[1,2,3], [4,5,6]],
                   [[1,2,3], [4,5,6]],
                   [[1,2,3], [4,5,6]]]).float()

x = torch.eye(3)
x = x.reshape((1, m, 3))
t2 = x.repeat(batch, 1, 1)

print(t1)
print(t1.shape)
print(t2)
print(t2.shape)
out_bmm = torch.bmm(t1, t2)

print(out_bmm)

# Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
z = x1 ** x2

# Other useful operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x,y)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0, max=10)

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)
z = torch.all(x)


