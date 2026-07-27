import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from classification_config import *
from classification_data import *
from classification_engine import *
from classification_model import *

def test_new_data(model,test_loader):
    # 取一个批次的测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # 推理预测
    with torch.no_grad():
        inputs = images.to(device)
        # 前向传播
        outputs = model(inputs)

    # 得到分类标签
    pred_labels = outputs.argmax(dim=1).cpu().numpy()
    # print(pred_labels.shape)

    # 转换输入图片,为画图做准备
    images = images.permute(0, 2, 3, 1).numpy()
    # print(images.shape)

    # 对比显示预测结果
    fig, axes = plt.subplots(1, 10, figsize=(25, 4), sharey=True, sharex=True)
    for i in range(10):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        # 真实标签
        print(f"{i + 1}-label: {labels[i]}")
        # 预测标签
        print(f"{i + 1}-pred: {pred_labels[i]}, 分类名: {CLASSIFICATION_NAMES[pred_labels[i]]}")
        print()
    plt.show()


if __name__ == '__main__':
    # 设置随机数种子
    seed_everything(SEED)
    # 加载模型
    model = ClassificationModel()
    model.load_state_dict(torch.load(CLASSIFIER_MODEL_NAME))
    model = model.to(device)
    print("=====模型加载完成=====")
    # 加载数据
    _, _, test_dataset = create_datasets()
    print("=====数据集创建完成=====")
    # 构建加载器
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)
    print("=====数据加载器创建完成=====")

    # 测试
    test_new_data(model,test_loader)
    this_acc_num = test_data(model, test_loader)

    print(f"准确率: {this_acc_num}")