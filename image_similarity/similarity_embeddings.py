import os
import torchvision.transforms as T
import numpy as np
import torch
from PIL import Image
from chromadb import EmbeddingFunction
from chromadb.api.types import Images, Embeddings
from tqdm import tqdm

from common.utils import sorted_alphanum
import chromadb
from image_similarity.similarity_model import ConvEncoder
from similarity_config import *


# 自定义的嵌入函数
class ImageEmbeddingFunction(EmbeddingFunction[Images]):
    # 传入自己的嵌入模型
    def __init__(self, model) -> None:
        self.model = model.to('cpu')
        return

    def __call__(self, input: Images) -> Embeddings:
        # 将输入图像转换为tensor
        input_tensor = torch.tensor(np.array(input))
        # 前向传播
        with torch.no_grad():
            embeddings = self.model(input_tensor)
        # 转成ndarray返回
        return embeddings.numpy()


# 加载全部图片,返回字典{id,image}
def get_id2images(image_dir, transform):
    id2images = {}
    # 读取目录下所有图片文件名
    image_names = sorted_alphanum(os.listdir(image_dir))
    # 遍历每个文件名,打开图片进行转换
    with tqdm(total=len(image_names), desc="图片加载中...") as pbar:
        for id, image_name in enumerate(image_names):
            # 1. 图片的完整访问路径
            image_path = os.path.join(image_dir, image_names[id])
            # 2. 打开图片
            image = Image.open(image_path).convert("RGB")
            # 3. 应用转换操作,得到tensor
            if transform is not None:
                image_tensor = transform(image)
            else:
                raise ValueError("transform 参数不能为None!")
            id2images[str(id)] = image_tensor.numpy()

            # 进度条更新
            pbar.update(1)

    return id2images


# 获取chroma的集合
def get_chroma_collection(encoder):
    # 1. 创建客户端
    path = os.path.join('..', PACKAGE_NAME, CHROMA_BACKEND_PATH)
    client = chromadb.PersistentClient(path=path)
    # 2. 场景集合
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ImageEmbeddingFunction(encoder),
    )
    return collection


# 生成所有图像的嵌入向量 (预处理)
def create_embeddings(encoder):
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
    ])
    print("=====开始生成所有图像的嵌入向量=====")
    # 1. 加载全部图片
    id2images = get_id2images(IMG_PATH, transform)
    print("=====图片加载完成=====")
    ids = list(id2images.keys())
    images = list(id2images.values())
    # 2. 获取chroma的集合
    collection = get_chroma_collection(encoder)
    # 3. 执行写入chroma写入操作
    print("=====开始写入chroma=====")
    # 分批写入
    batch = np.ceil(len(ids) / CHROMA_INSERT_BATCH_SIZE)
    for i in range(int(batch)):
        start = i * CHROMA_INSERT_BATCH_SIZE
        end = min((i + 1) * CHROMA_INSERT_BATCH_SIZE, len(ids))
        collection.upsert(
            ids=ids[start:end],
            images=images[start:end],
        )
    print("=====写入完成=====")


# 相似图片搜索
def search_similar_image_ids(collection, image, cnt=5):
    result = collection.query(
        query_images=image.numpy(),
        n_result=cnt
    )
    similar_image_ids = [int(id) for id in result['ids'][0]]
    return similar_image_ids


if __name__ == '__main__':
    encoder = ConvEncoder()
    encoder.load_state_dict(torch.load(ENCODER_MODEL_NAME))
    create_embeddings(encoder)
