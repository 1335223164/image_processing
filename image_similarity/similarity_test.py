import torch
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from similarity_embeddings import *
from similarity_config import *
from similarity_data import *
from similarity_engine import *
from similarity_model import *

if __name__ == '__main__':
    # 设置随机数种子
    seed_everything(SEED)
    # 加载模型
    encoder = ConvEncoder()
    encoder.load_state_dict(torch.load(ENCODER_MODEL_NAME))
    encoder = encoder.to(device)
    print("=====模型加载完成=====")
    # 加载数据
    _, _, test_dataset = create_dataset()
    print("=====数据集创建完成=====")
    # 从测试集获取一张新图片
    image, _ = test_dataset[0]
    print(image.shape)

    # 获取chroma集合
    collection = get_chroma_collection(encoder)

    # 测试
    similar_image_ids = search_similar_image_ids(collection, image, cnt=5)
    print(similar_image_ids)

    # 画图
    fig, axes = plt.subplots(2, 5, figsize=(25, 4))
    # 输入图片
    image = image.permute(1, 2, 0).cpu().numpy()
    axes[0, 2].imshow(image)
    # 相似图片
    for i in range(len(similar_image_ids)):
        # 拼接文件名
        image_name = str(similar_image_ids[i]) + ".jpg"
        # 读取图片
        image = Image.open(os.path.join(IMG_PATH, image_name)).convert("RGB")
        # 画图显示
        axes[0, i].imshow(image)

    for ax in axes.flat:
        ax.axis('off')

    plt.show()
