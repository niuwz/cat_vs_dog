import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from my_models import *
from cat_dog_dataset import my_Dataset


def train(epoch, model, train_dataloader, criterion, optimizer):
    '''模型训练函数'''
    global print_loss
    # 调整为训练模式
    model.train()
    losses = []
    plot_loss = 0
    total_num = len(train_dataloader.dataset)
    print_thre = total_num//(BATCH_SIZE*5)
    for i, (img, label) in enumerate(train_dataloader):
        img, label = img.to(DEVICE), label.to(DEVICE)
        out = model(img)
        # 计算损失
        loss = criterion(out, label.squeeze())
        losses.append(loss.item())
        # 误差反向传播
        loss.backward()
        # 优化采用设定的优化方法对网络中的各个参数进行调整
        optimizer.step()
        # 清除优化器中的梯度以便下一次计算
        optimizer.zero_grad()
        if (i+1) % print_thre == 0:
            plot_loss += np.mean(losses)
            print('TRAINING: Epoch:{} [{} / {} ({:.2f}%)] Loss:{}'.format(
                epoch, (i+1)*BATCH_SIZE, total_num, (i+1)*BATCH_SIZE/total_num*100, plot_loss))
            losses.clear()
    print_loss.append(plot_loss/5)


def test(model, test_dataloader, criterion):
    '''模型验证函数'''
    model.eval()
    cnt = 0
    losses = []
    total_num = len(test_dataloader.dataset)
    for i, (img, label) in enumerate(test_dataloader):
        img, label = img.to(DEVICE), label.to(DEVICE)
        out = model(img)
        # 计算损失
        loss = criterion(out, label.squeeze())
        losses.append(loss.item())
        y = out.argmax(1, keepdim=True)
        # 计算错误个数
        num = (y-label).abs().sum()
        cnt += num
    # 计算准确率
    acc = 1-cnt/total_num
    print('\nTESTING: Loss:{},Accuracy:[{} / {} ({:.6f}%)]\n'.format(np.mean(losses), total_num-cnt, total_num,
          acc*100))
    return acc.item()


if __name__ == "__main__":
    global PATH, IMG_SIZE, BATCH_SIZE, DEVICE
    '''定义相关超参数'''
    # 训练集路径
    PATH = "data/train"
    # 裁剪图像大小
    IMG_SIZE = 256
    BATCH_SIZE = 25
    EPOCHS = 5
    DEVICE = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    WORKERS = 0
    # 训练数据比例
    TRAIN_RATE = 0.7

    print("using device {}".format(DEVICE))

    # 数据集分割，分为训练集和验证集，采用固定的seed
    img_data = my_Dataset("train", PATH, IMG_SIZE)
    IMG_NUM = len(img_data)
    TRAIN_NUM = int(IMG_NUM*TRAIN_RATE)
    train_data, vali_data = data.random_split(
        img_data, [TRAIN_NUM, IMG_NUM-TRAIN_NUM], torch.Generator().manual_seed(1))
    train_dataloader = DataLoader(train_data, BATCH_SIZE,
                                  shuffle=True, num_workers=WORKERS, drop_last=True)
    vali_dataloader = DataLoader(vali_data, BATCH_SIZE,
                                 shuffle=True, num_workers=WORKERS, drop_last=True)

    # 定义网络模型
    # ResNet18模型
    model = get_resnet18_model(False, True).to(DEVICE)
    # VGG16模型
    # model = get_vgg16_model(False, True).to(DEVICE)
    # CNN模式
    # model = get_cnn_model(False).to(DEVICE)
    model_name = model._get_name()
    print("using model:", model_name)
    # 实例化一个优化器，优化方式为adam方法
    optimizer = optim.Adam(model.parameters())
    # 定义损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()

    print_loss = []
    accs = []
    # 训练过程
    for epoch in range(EPOCHS):
        train(epoch, model, train_dataloader, criterion, optimizer)
        # 验证集准确率
        acc = test(model, vali_dataloader, criterion)
        accs.append(acc)
    if model_name == "VGG":
        torch.save(
            model.classifier._modules["6"].state_dict(), model_name+"_state.pth")
        print(model_name+"模型参数保存成功")
    else:
        torch.save(model.state_dict(), model_name+"_state.pth")
        print(model_name+"模型参数保存成功")

    plt.figure(1)
    plt.plot(print_loss)
    plt.xlabel("EPOCHS")
    plt.ylabel("Loss")
    plt.title(model_name+" Loss下降曲线")
    plt.savefig("Loss_{}.jpg".format(model_name))

    plt.figure(2)
    plt.plot(accs)
    plt.xlabel("EPOCHS")
    plt.ylabel("Accuracy")
    plt.title(model_name+" 准确率变化情况")
    plt.savefig("Accuracy_{}.jpg".format(model_name))
    plt.show()
