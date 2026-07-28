# 数据预处理
IMG_PATH = '../common/dataset/'
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
PACKAGE_NAME = 'image-similarity'
ENCODER_MODEL_NAME = "encoder.pt"
DECODER_MODEL_NAME = "decoder.pt"

# 向量数据库的配置
CHROMA_BACKEND_PATH = 'chroma_backend'
CHROMA_COLLECTION_NAME = 'image_collection'
CHROMA_INSERT_BATCH_SIZE = 5000