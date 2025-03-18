import argparse
import torch
import pickle
import warnings
import numpy as np
from model import Vulsim_detector

warnings.filterwarnings("ignore")  # 忽视警告

# 设置随机种子
seed = 71926                       # 生成随机数种子的初值
np.random.seed(seed)               # 为numpy随机数生成器设置随机数种子
torch.manual_seed(seed)            # 为cpu的随机数生成器设置随机数种子
torch.cuda.manual_seed(seed)       # 为当前cuda的随机数生成器设置随机数种子
torch.cuda.manual_seed_all(seed)   # 为所有cuda的随机数生成器设置随机数种子

def parse_options():
    parser = argparse.ArgumentParser(description='VulEGRL training.')
    parser.add_argument('-tr', '--train_path', help='The path of train.pkl', type=str, default='./dataset/train_dataset.pkl')
    parser.add_argument('-te', '--test_path', help='The path of test.pkl', type=str, default='./dataset/test_dataset.pkl')
    # parser.add_argument('-bg', '--bg_path', help='The path of bg_G.pkl', type=str, default='../preprocess/dataset/slice_data/bg/FFMpeg+Qemu/bg_G.pkl')
    args = parser.parse_args()
    return args

def main():
    args = parse_options()
    train_path = args.train_path  # 训练集路径
    test_path = args.test_path  # 测试集路径 

    node_num = 100   # 每个切片的结点数
    input_dim = 128  # 输入向量维度
    output_dim = 128 # 输出向量维度 
    epochs = 100
    batch_size = 64 # 最好数据，batch_size=32
    learning_rate = 0.0001
    decay = 0.001
    dropout= 0.5

    print("正在读取数据...")
    with open(train_path,'rb') as f:  # 读取数据
        train_data = pickle.load(f)
    with open(test_path,'rb') as f:  # 读取数据
        test_data = pickle.load(f)

    detector = Vulsim_detector(node_num, input_dim, output_dim, epochs, batch_size, \
                 learning_rate, decay, dropout)
    detector.preparation(train_data, test_data)
    detector.train()

if __name__ == "__main__":
    main()