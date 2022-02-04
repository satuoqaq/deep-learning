import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # load image
    img_path = "img.png"
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # plt.show()

    # N,Channel,Height,Width
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = 'class_indices.json'
    assert os.path.exists(json_path), "file '{}' does not exist.".format(json_path)
    class_indict = json.load(open(json_path, "r"))

    # create model
    model = AlexNet(num_classes=5).to(device)

    # load model weight
    weights_path = "AlexNet.pth"
    assert os.path.exists(weights_path), "file '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    # not use BP and Dropout
    model.eval()

    # test
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class :{}   prob:{:.3}".format(class_indict[str(predict_cla)],
                                                predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class :{:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == "__main__":
    main()
