import numpy as np
import torch
import os
import random
import re

# 实现一个统一设置随机数种子的函数,消除随机性
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # pytorch设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sorted_alphanum(img_names):
    convert = lambda str: int(str) if str.isdigit() else str.lower()
    alphanum_key = lambda x: [convert(c) for c in re.split(r'([0-9]+)', x)]
    return sorted(img_names, key=alphanum_key)





if __name__ == '__main__':
    img_names = ["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg","6.jpg","7.jpg","8.jpg","9.jpg","10.jpg"]
    sorted_img_names = sorted_alphanum(img_names)
    print(sorted_img_names)