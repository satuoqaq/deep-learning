import torch
import torch.nn as nn


class FasterRCNNBase(nn.Module):
    def __init__(self):
        super(FasterRCNNBase, self).__init__()
        self.x = 1
