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
NOISE_FACTOR = 0.5

# 训练超参数
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 30

# 项目配置
PACKAGE_NAME = 'image-denoising'
DENOISER_MODEL_NAME = "denoiser.pt"