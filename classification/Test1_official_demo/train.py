import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 全局取消证书验证
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)
    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=5000,
                                              shuffle=False, num_workers=0)

    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # print labels
    # print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
    # show images
    # imshow(torchvision.utils.make_grid(test_image[0:4]))

    # 网络
    net = LeNet()
    # 损失
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # 图片和标签
            inputs, labels = data
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播+backward+optimize
            output = net(inputs)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(test_image)
                    predict_y = torch.max(outputs, dim=1)[1]
                    print(predict_y, torch.max(outputs, dim=1))
                    accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                    print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == "__main__":
    main()
