import torch

a = [[1, 10], [2, 3]]
a = torch.Tensor(a)
print(a)

print(a.max(dim=1)[0])
