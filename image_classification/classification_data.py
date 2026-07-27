import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, random_split
from common.utils import *
import torchvision.transforms as T
from image_classification.classification_config import *


class ImageLabelDataset(Dataset):
    # 初始化
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        label_data = pd.read_csv(label_path)
        self.labels = label_data['target'].tolist()
        self.image_names = sorted_alphanum(os.listdir(image_dir))

    # 获取数据集长度
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
        # 4. 找到图片对应的标签
        label = self.labels[index]
        return image_tensor, label

def create_datasets():
    # 定义图像转化操作
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
    ])
    # 创建数据集
    dataset = ImageLabelDataset(image_dir=IMG_PATH, label_path=LABEL_PATH, transform=transform)
    # 划分数据
    train_dataset, val_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, VAL_RATIO, TEST_RATIO])
    return train_dataset, val_dataset, test_dataset



if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = create_datasets()
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}")
