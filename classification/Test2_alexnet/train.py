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

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize(size=(224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}
print(os.getcwd())
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
print(data_root)
image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
print("image_path:", image_path)

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
train_num = len(train_dataset)
print(train_num)

flower_list = train_dataset.class_to_idx
print(flower_list)
# change class_to_idx to dict
cla_dict = dict((val, key) for key, val in flower_list.items())
print(cla_dict)
# write dict to json
# json_str = json.dumps(cla_dict, indent=4)
# with open('class_indices.json', 'w') as json_file:
#     json_file.write(json_str)

batch_size = 32
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
print(val_num)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)
print("Using {} images for training ,{} images for validation.".format(train_num, val_num))

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))


net = AlexNet(num_classes=5, init_weights=True)
net.to(device)

# 损失函数
loss_function = nn.CrossEntropyLoss()
# 优化器 学习率
optimizer = optim.Adam(net.parameters(), lr=0.0002)

epochs = 10
save_path = 'AlexNet.pth'
best_acc = 0.0
train_steps = len(train_loader)
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        # 计算损失
        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

    # print()
    # print(time.perf_counter() - t1)

    # validate 验证准确率
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        print('[epoch %d] train_loss : %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, accurate_test))

        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)

print('Finished Training')
