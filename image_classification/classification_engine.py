from tqdm import tqdm
from classification_data import *

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练一个伦次
def train_one_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total = 0
    for input, target in train_loader:
        input = input.to(device)
        target = target.to(device)
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
    for input, target in val_loader:
        input = input.to(device)
        target = target.to(device)
        # 前向传播
        output = model(input)
        # 计算损失
        loss = loss_fn(output, target)
        total_loss += loss.item() * input.shape[0]
    # 计算平均损失
    this_loss = total_loss / len(val_loader.dataset)
    return this_loss

# 测试
def test_data(model, test_loader):
    model.eval()
    total_num = 0
    test_acc_num = 0
    for input, target in tqdm(test_loader, desc="测试中..."):
        input = input.to(device)
        target = target.to(device)
        # 前向传播
        output = model(input)
        # 得到预测分类号
        pred_label = output.argmax(dim=1)
        # 累加准确个数
        test_acc_num += (pred_label == target).sum().item()
        total_num += input.shape[0]
    # 返回准确率
    this_acc_num = test_acc_num / total_num
    return this_acc_num