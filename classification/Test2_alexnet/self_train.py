import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet
import json
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# data_transform
data_transform = {
    "train": transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val": transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
image_path = os.path.join(data_root, "data_set", "flower_data")

# 指定数据集
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"))
val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"))
# 加载数据集
batch_size = 32
num_workers = 0

# 训练测试集数量
train_num = len(train_dataset)
val_num = len(val_dataset)
train_loader = torch.utils.data.Loader(train_dataset,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       shuffle=True)
val_loader = torch.utils.data.Loader(val_dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=True)
# 花的字典写入json, 1:'rose' 的形式
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
with open('class_indices.json', 'w') as json_file:
    json_file.write(json.dumps(cla_dict, indent=4))

# 网络
net = AlexNet(num_classes=5, init_weights=True)
net.to(device)
# 损失函数
loss_function = nn.CrossEntropyLoss()
# 优化器 学习率
optimizer = optim.Adam(net.parameters(), lr=0.0002)

epochs = 10
save_path = 'AlexNet.pth'
best_acc = 0.0
train_steps = len(train_dataset)

for epoch in epochs:

    # 训练集训练
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        images, labels = data
        # 每一个Batch梯度都先清零
        optimizer.zero_grad()
        # 丢到网络里边
        outputs = net(images.to(device))
        # 计算损失
        loss = loss_function(outputs, labels.to(device))
        # 反向传播
        loss.backward()
        # 一次自定义学习率的梯度下降
        optimizer.step()
        # 计算损失和
        running_loss += loss.item()
        train_bar.desc = "train epoch [{}/{}] loss:{:.3f}".format(epoch + 1, epochs, running_loss)

    # 测试集算精度

    # net.eval() 表示不使用BP和Dropout
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for step, data in val_loader:
            # 取出验证集图片与标签
            images, labels = data
            # 计算一下返回值
            outputs = net(images.to(device))
            # 计算准确率
            # output 返回的是每一张图片,对应每种花的概率
            # 如果选择dim=0,返回的是每种花最高可能图片下标
            # 选择dim=1,返回的每张图片最可能花的下标
            # torch.max 的返回值和下标,要的是下标,
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq()
