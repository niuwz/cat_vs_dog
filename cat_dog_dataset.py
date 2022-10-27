import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class my_Dataset(data.Dataset):
    def __init__(self, mode, path, IMG_SIZE) -> None:
        '''
        mode: 1. train 返回数据和标签，用于训练和验证 2. test 返回文件名和数据，用于测试
        path: 数据集所在路径
        IMG_SIZE: 裁剪后的图像大小，这里使用256*256
        '''
        super(my_Dataset, self).__init__()
        self.mode = mode
        self.size = 0
        self.img_names = []
        self.img_labels = []
        self.path = path
        # 数据处理
        self.dataTransform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])
        self.__img_init()

    def __img_init(self):
        '''此处生成文件名列表，取出数据时再读取文件'''
        if self.mode == "train":
            types = ["cat", "dog"]
            for ilabel in range(len(types)):
                path = self.path+"/"+types[ilabel]
                temp_list = os.listdir(path)
                self.img_names += temp_list
                self.img_labels += [ilabel]*len(temp_list)
                self.size += len(temp_list)
        elif self.mode == "test":
            temp_list = os.listdir(self.path)
            self.img_names += temp_list
            self.size += len(temp_list)

    def __getitem__(self, index):
        '''
        读取数据
        train模式: 返回两个tensor, 分别为模型输入和标签
        test模式: 返回图像文件名和模型输入tensor
        '''
        types = ["cat", "dog"]
        if self.mode == "train":
            y = self.img_labels[index]
            img = Image.open(self.path+"/"+types[y]+"/"+self.img_names[index])
            x = self.dataTransform(img)
            y = torch.tensor([y])
            return x, y
        elif self.mode == "test":
            name = self.img_names[index]
            img = Image.open(self.path+"/"+name)
            x = self.dataTransform(img)
            return name, x

    def __len__(self):
        return self.size
