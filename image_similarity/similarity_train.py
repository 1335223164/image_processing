import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from similarity_config import *
from similarity_data import *
from similarity_engine import *
from similarity_model import *

if __name__ == '__main__':
    # 设置随机数种子
    seed_everything(SEED)
    # 加载模型
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    # 加载数据
    train_dataset, val_dataset, _ = create_dataset()
    print("=====数据集创建完成=====")

    # 构建加载器
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, drop_last=True)
    print("=====数据加载器创建完成=====")

    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=LEARNING_RATE)

    # 训练流程
    min_val_loss = float('inf')
    patience = 3  # 容忍多少次验证损失不下降
    counter = 0  # 计数器
    print("=====开始训练=====")
    for epoch in tqdm(range(EPOCHS)):
        # 训练
        this_loss = train_one_epoch(encoder, decoder, train_loader, optimizer, loss_fn)
        # 验证
        val_loss = val_step(encoder, decoder, val_loader, loss_fn)
        print(f"epoch: {epoch + 1}, loss: {this_loss}, val_loss: {val_loss}")

        # 判断如果验证损失减小,就保存模型
        if val_loss < min_val_loss:
            print("验证损失减小,保存模型...")
            min_val_loss = val_loss
            counter = 0  # 重置计数器
            torch.save(encoder.state_dict(), ENCODER_MODEL_NAME)
        else:
            counter += 1
            print(f"验证损失没有减小 ({counter}/{patience})...")
            if counter >= patience:
                print("早停条件触发，停止训练")
                break
    print("=====训练结束=====")
