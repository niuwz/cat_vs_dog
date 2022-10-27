import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from torch import nn
from cat_dog_dataset import my_Dataset
from my_models import *


if __name__ == "__main__":
    PATH = "data/test"
    SHOW_NUM = 3
    BATCH_SIZE = 9
    IMG_SIZE = 256
    TYPES = ["cat", "dog"]
    DEVICE = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_data = my_Dataset("test", PATH, IMG_SIZE)
    test_dataloader = DataLoader(img_data, BATCH_SIZE, shuffle=True)

    model = get_resnet18_model(True).to(DEVICE)
    # model = get_vgg16_model(True).to(DEVICE)

    model.eval()
    for i, (names, x) in enumerate(test_dataloader):
        if i >= SHOW_NUM:
            break
        plt.figure()
        x = x.to(DEVICE)
        out = model(x)
        prediction = out.cpu().detach().numpy()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.3, hspace=0.5)
        for idx in range(BATCH_SIZE):
            plt.subplot(3, 3, idx+1)
            name = names[idx]
            result = prediction[idx]
            img = plt.imread(PATH+"/"+name)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.title("[cat:{:.1f}%,dog:{:.1f}%]".format(
                result[0]*100, result[1]*100))
    plt.show()
