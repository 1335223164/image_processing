import torch
from torch import nn, optim

# 分别定义编码器和解码器类
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        # 通用池化层
        self.pool = nn.MaxPool2d(2, 2)
    # 前向传播
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        x = torch.relu(self.conv6(x))
        # 压缩成向量形式
        x = x.squeeze(-1).squeeze(-1)
        return x

class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        # 卷积层
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
    # 前向传播
    def forward(self,x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        return x

if __name__ == '__main__':
    input = torch.randn(10, 3, 64, 64)
    encoder = ConvEncoder()
    decoder = ConvDecoder()
    # 前向传播
    embeddings = encoder(input)
    print(embeddings.shape)
    output = decoder(embeddings)
    print(output.shape)