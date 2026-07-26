import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from denoising_config import *
from denoising_data import *

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练一个伦次
def train_one_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total = 0
    for noise_img, image in train_loader:
        input = noise_img.to(device)
        target = image.to(device)
        # 前向传播
        output = model(input)
        # 计算损失
        loss = loss_fn(output, target)
        # 反向传播 + 梯度清零 + 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 累加损失
        total_loss += loss.item()
        total += 1
    # 本轮训练结束,计算平均损失
    this_loss = total_loss / total
    return this_loss

# 验证一个伦次
def val_step(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total = 0
    for noise_img, image in val_loader:
        input = noise_img.to(device)
        target = image.to(device)
        # 前向传播
        output = model(input)
        # 计算损失
        loss = loss_fn(output, target)
        total_loss += loss.item()
        total += 1
    # 本轮训练结束,计算平均损失
    this_loss = total_loss / total
    return this_loss

# 测试
def test_data(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total = 0
    for noise_img, image in tqdm(test_loader, desc="测试中..."):
        input = noise_img.to(device)
        target = image.to(device)
        # 前向传播
        output = model(input)
        # 计算损失
        loss = loss_fn(output, target)
        total_loss += loss.item()
        total += 1
    # 本轮训练结束,计算平均损失
    this_loss = total_loss / total
    return this_loss