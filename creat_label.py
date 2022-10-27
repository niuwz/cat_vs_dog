import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cat_dog_dataset import my_Dataset
from my_models import *

if __name__ == "__main__":
    global PATH, IMG_SIZE, DEVICE
    PATH = "data/test"
    target_path = "data/result"
    IMG_SIZE = 256
    TYPES = ["cat", "dog"]
    DEVICE = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 实例化测试数据集
    img_data = my_Dataset("test", PATH, IMG_SIZE)
    test_dataloader = DataLoader(img_data)
    print("using device", DEVICE)
    # 加载训练好的模型
    # model = get_vgg16_model(True).to(DEVICE)
    model = get_resnet18_model(True).to(DEVICE)
    print("using model:", model._get_name())

    # 收集文件名和类别
    file_names_list = []
    type_list = []

    model.eval()
    for i, (names, x) in enumerate(test_dataloader):
        x = x.to(DEVICE)
        name = names[0]
        out = model(x)
        idx = out.argmax(1, keepdim=True)
        file_names_list.append(name)
        type_list.append(idx.item())
        out_type = TYPES[idx]
        # 转存图片
        img = Image.open(PATH+"/"+name)
        img.save(target_path+"/"+out_type+"/"+name)
        img.close()
    print("分类完毕")
    result_dict = {
        "file_name": file_names_list,
        "type": type_list
    }
    result = pd.DataFrame(result_dict)
    result.to_excel(model._get_name()+"_result.xlsx")
    print("类别文件生成完毕")
