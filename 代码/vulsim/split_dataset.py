import tqdm
import random
import numpy as np
import pickle

def data_split(data, ratio): # 分别对有漏洞的数据集和无漏洞数据集进行切分

    train_dataset = []  # 训练集
    test_dataset = []   # 测试集
    total = len(data)
    offset = int(total*ratio)
    index_list = np.arange(total)
    random.shuffle(index_list)         # 打乱顺序
    train_index = index_list[:offset]  # 训练集索引
    test_index = index_list[offset:]   # 测试集索引
    
    for i in train_index:
        train_dataset.append(data[i])
    for i in test_index:
        test_dataset.append(data[i])

    return train_dataset, test_dataset

def main():
    dataset_path = './dataset/forward+backward/func_embedding.pkl'  # 切片路径
    train_path = './dataset/forward+backward/train_dataset.pkl'  # 训练集保存路径
    test_path = './dataset/forward+backward/test_dataset.pkl'    # 测试集保存路径
    
    ratio = 0.8 # 切分比例
    Fun_0_data = [] # 保存无漏洞函数
    Fun_1_data = [] # 保存有漏洞函数

    with open(dataset_path,'rb') as f:
        funcs_info =pickle.load(f)
    
    print("正在收集有漏洞切片和无漏洞切片...")
    for func in tqdm.tqdm(funcs_info):
        label = funcs_info[func]['label'] # 切片标签
        if label == 0: # 无漏洞切片
            Fun_0_data.append(funcs_info[func])
        elif label == 1: # 有漏洞切片
            Fun_1_data.append(funcs_info[func])

    print("正在切分数据集...")
    train_0_dataset,test_0_dataset = data_split(Fun_0_data,ratio)  # 对无漏洞函数进行切分
    train_1_dataset,test_1_dataset = data_split(Fun_1_data,ratio)  # 对有漏洞函数进行切分

    print("打乱数据集各元素位置...")
    train_dataset = [j for i in [train_0_dataset,train_1_dataset] for j in i]
    test_dataset = [j for i in [test_0_dataset,test_1_dataset] for j in i]
    random.shuffle(train_dataset)  # 打乱顺序
    random.shuffle(test_dataset)

    print("开始保存训练集数据至于：", train_path)
    with open(train_path, 'wb') as f:
        pickle.dump(train_dataset, f)

    print("开始保存训测试集数据至于：", test_path)   
    with open(test_path, 'wb') as f:
        pickle.dump(test_dataset, f)  

if __name__ == "__main__":
    main()