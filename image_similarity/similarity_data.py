import os
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from similarity_config import *
from common.utils import *
import torchvision.transforms as T


class ImageDataset(Dataset):
    # 初始化
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(image_dir))

    # 获取数据集大小
    def __len__(self):
        return len(self.image_names)

    # 获取数据集的某一个数据
    def __getitem__(self, index):
        # 1. 图片的完整访问路径
        image_path = os.path.join(self.image_dir, self.image_names[index])
        # 2. 打开图片
        image = Image.open(image_path).convert("RGB")
        # 3. 应用转换操作,得到tensor
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            raise ValueError("transform 参数不能为None!")

        return image_tensor, image_tensor


# 创建数据集并划分
def create_dataset():
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
    ])
    # 创建数据集
    dataset = ImageDataset(image_dir=IMG_PATH, transform=transform)
    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO])

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    # dataset = NoiseImageDataset(image_dir=IMG_PATH)
    train_dataset, val_dataset, test_dataset = create_dataset()
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}")
