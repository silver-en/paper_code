import os
import pickle
import numpy as np
import tqdm

if __name__ == '__main__':
    path = "./dataset/adj/forward+backward"
    info_path = "../embedding/vulsim/forward+backward/forward+backward_No_Vul.pkl" # 文件保存路径
    FUN_info = {} # 保存所有数据的信息
    node_num = 100 # 节点个数

    # for type in ['Vul','No_Vul']:
    for dir in tqdm.tqdm(os.listdir(path+'/No_Vul')):

        func_info = {} # 收集单个函数的所有信息
        fun_path = path+'/No_Vul/'+dir
        with open(fun_path,'rb') as f:
            data = pickle.load(f)

        func_info['code_tokens'] = np.array(data['code_tokens'])[:node_num] # 切片序列信息
        func_info['CDG_adj'] = np.array(data['CDG_adj'])[:node_num,:node_num] # CDG拓扑结构信息
        func_info['DDG_adj'] = np.array(data['DDG_adj'])[:node_num,:node_num] # DDG拓扑结构信息
        func_info['PDG_adj'] = np.array(data['PDG_adj'])[:node_num,:node_num] # PDG拓扑结构信息
        func_info['label'] = data['label']# 当前函数的标签

        FUN_info[dir] = func_info

    with open(info_path, 'wb') as f:
        pickle.dump(FUN_info, f)
            