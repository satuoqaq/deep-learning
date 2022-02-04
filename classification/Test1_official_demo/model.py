import torch.nn as nn
import torch.nn.functional as F
import torch


class LeNet(nn.Module):
    def __init__(self):
        # 多继承
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))  # input(3,32,32) out(16,28,28)
        X = self.pool1(X)  # out(16,14,14)
        X = F.relu(self.conv2(X))  # out(32,10,10)
        X = self.pool2(X)  # out(32,5,5)
        # -1 表示自动推理,给x展平
        X = X.view(-1, 32 * 5 * 5)  # out(32*5*5)
        X = F.relu(self.fc1(X))  # out(120)
        X = F.relu(self.fc2(X))  # out(84)
        X = self.fc3(X)  # out(10)
        return X


def main():
    input1 = torch.rand([32, 3, 32, 32])
    model = LeNet()
    # print(model)
    output = model(input1)


if __name__ == "_main__":
    main()
