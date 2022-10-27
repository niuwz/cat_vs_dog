import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # 读取文件
    PATH = "data/test/"
    TYPES = ["cat", "dog"]
    vgg_data = pd.read_excel("VGG_result.xlsx")
    resnet_data = pd.read_excel("resnet_result.xlsx")
    # 计算相似程度
    n = len(vgg_data)
    result = (vgg_data["type"] == resnet_data["type"])
    num = result.sum()
    print("vgg16模型与ResNet18模型分类结果相似度为:{}%".format(num/n*100))
    # 展示分类不同的图片
    for i in range(n):
        if not result[i]:
            img = Image.open(PATH+vgg_data["file_name"][i])
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.title("VGG16:{} ResNet18:{}".format(
                TYPES[vgg_data["type"][i]], TYPES[resnet_data["type"][i]]))
            plt.show()
