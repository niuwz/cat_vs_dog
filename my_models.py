
from cnn_model import Net
import torch
from torch import nn
from torchvision import models


def get_cnn_model(load_state):
    '''
    load_state: 是否加载模型参数
    '''
    model = Net()
    # 加载模型参数
    if load_state:
        cnn_state = torch.load("cnn_state.pth")
        model.load_state_dict(cnn_state, strict=False)
    return model


def get_vgg16_model(load_state, pretrained=False):
    '''
    load_state: 是否加载模型参数
    pretrained: 是否使用预训练模型参数
    '''
    vgg16_model = models.vgg16(pretrained)
    for param in vgg16_model.parameters():
        # 锁定预训练参数
        param.requires_grad = False
    # 更改最后一层全连接层
    vgg16_model.classifier._modules['6'] = nn.Linear(4096, 2)
    vgg16_model.classifier._modules['7'] = nn.Softmax(1)
    # 加载模型参数
    if load_state:
        vgg_state = torch.load("vgg_model_state.pth")
        vgg16_model.classifier._modules['6'].load_state_dict(vgg_state)
    return vgg16_model


def get_resnet18_model(load_state, pretrained=False):
    '''
    load_state: 是否加载模型参数
    pretrained: 是否使用预训练模型参数
    '''
    resnet18 = models.resnet18(pretrained)
    for paras in resnet18.parameters():
        # 锁定预训练参数
        paras.requires_grad = False
    # 更改最后一层全连接层
    resnet18.fc = nn.Sequential(nn.Linear(512, 2), nn.Softmax(1))
    # 加载模型参数
    if load_state:
        resnet18_state = torch.load("resnet18_state.pth")
        resnet18.load_state_dict(resnet18_state, strict=False)
    return resnet18
