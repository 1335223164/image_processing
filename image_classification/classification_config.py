# 数据预处理
IMG_PATH = '../common/dataset/'
LABEL_PATH = '../common/fashion-labels.csv'
IMG_H = 64
IMG_W = 64

# 随机性相关配置
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# 训练超参数
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 20

# 项目配置
PACKAGE_NAME = 'image-classification'
CLASSIFIER_MODEL_NAME = "classification.pt"

# 定义标签和作文名称对应关系
CLASSIFICATION_NAMES = {
    0: '上衣',
    1: '鞋',
    2: '包',
    3: '下装',
    4: '手表'
}