from torch import nn


class ClassificationModel(nn.Module):
    def __init__(self, n_classes=5):
        super(ClassificationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(1),
            nn.Linear(16 * 16 * 16, n_classes),
        )

    # 前向传播
    def forward(self, x):
        return self.model(x)

