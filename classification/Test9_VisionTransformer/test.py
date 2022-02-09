from typing import Tuple

import numpy as np

import torch
from torch import Tensor

a = torch.randn(60, 60,22, 3)  # 输入的维度是（60，30）
b = torch.nn.Linear(3, 9)  # 输入的维度是（30，15）
#
output = b(a)
print('b.weight.shape:\n ', b.weight.shape)
print('b.bias.shape:\n', b.bias.shape)
print('output.shape:\n', output.shape)
