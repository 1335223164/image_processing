import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from denoising_config import *
from denoising_data import *
from denoising_engine import *
from denoising_model import *

def test_new_data(model,test_loader):
    # 取一个批次的测试数据
    data_iter = iter(test_loader)
    noise_imgs, images = next(data_iter)

    # 推理预测
    with torch.no_grad():
        inputs = noise_imgs.to(device)
        # 前向传播
        outputs = model(inputs)

    # 转成ndarray,方便画图
    images_numpy = images.permute(0, 2, 3, 1).cpu().numpy()
    noise_img_numpy = noise_imgs.permute(0, 2, 3, 1).cpu().numpy()
    output_numpy = outputs.permute(0, 2, 3, 1).cpu().numpy()

    # 画图
    fig, axes = plt.subplots(3, 10, figsize=(24, 4), sharey=True, sharex=True)
    for ax_row, images in zip(axes, [images_numpy, noise_img_numpy, output_numpy]):
        for ax, img in zip(ax_row, images):
            ax.imshow(img)
            ax.set_axis_off()

    plt.show()


if __name__ == '__main__':
    # 设置随机数种子
    seed_everything(SEED)
    # 加载模型
    model = ConvDenoiser()
    model.load_state_dict(torch.load(DENOISER_MODEL_NAME))
    model = model.to(device)
    print("=====模型加载完成=====")
    # 加载数据
    _, _, test_dataset = create_dataset()
    print("=====数据集创建完成=====")
    # 构建加载器
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)
    print("=====数据加载器创建完成=====")

    # 测试
    test_new_data(model,test_loader)
    test_loss = test_data(model, test_loader, loss_fn=nn.MSELoss())

    print(f"测试损失: {test_loss}")