# 以每一行代码为单位，对切片代码进行向量化
import os
import re
import time
import tqdm
import pickle
import numpy as np
import glob
from gensim.models import Word2Vec

def tokenize_code_line(line):  # 切分token
    # Sets for operators
    operators3 = {'<<=', '>>='}
    operators2 = {
        '->', '++', '--', '!~', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||',
        '+=', '-=', '*=', '/=', '%=', '&=', '^=', '|='
    }
    operators1 = {
        '(', ')', '[', ']', '.', '+', '-', '*', '&', '/', '%', '<', '>', '^', '|',
        '=', ',', '?', ':', ';', '{', '}', '!', '~'
    }

    tmp, w = [], []
    i = 0
    if type(i) == None:
        return []
    while i < len(line):
        # Ignore spaces and combine previously collected chars to form words
        if line[i] == ' ':
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        # Check operators and append to final list
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 3].strip())
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 2].strip())
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        # Character appended to word list
        else:
            w.append(line[i])
            i += 1  
    if (len(w) != 0):
        tmp.append(''.join(w).strip())
        w = []

    # Filter out irrelevant strings
    tmp = list(filter(lambda c: (c != '' and c != ' '), tmp))
    return tmp

def splite_sentence(slices_path):
    sentence_list = []

    with open(slices_path,'rb') as f:
        slices_info = pickle.load(f)

    for slice in slices_info:
        # print(slices_info[slice]['slice_code'])
        for line in slices_info[slice]['slice_code']:
            tokens = tokenize_code_line(line.strip("\""))  # 分词
            sentence_list.append(tokens)

    return sentence_list

class IterCorpus():
    def __init__(self,No_func_path,Vul_func_path):
        self.no_funcs_path = No_func_path
        self.vul_funcs_path = Vul_func_path
    
    def __iter__(self):
        with open(self.no_funcs_path,'rb') as f:
            No_funcs_info = pickle.load(f)

        for func in No_funcs_info:
            # print(slices_info[slice]['slice_code'])
            
            for line in No_funcs_info[func]['code_tokens']:
                tokens = tokenize_code_line(line.strip("\""))  # 分词

                yield tokens
        
        with open(self.vul_funcs_path,'rb') as f:
            vul_funcs_info = pickle.load(f)
        for func in vul_funcs_info:
            # print(slices_info[slice]['slice_code'])
            # if vul_funcs_info[func]['code_tokens']:
            for line in vul_funcs_info[func]['code_tokens']:
                tokens = tokenize_code_line(line.strip("\""))  # 分词

                yield tokens


def main():
    No_func_path = "./vulsim/forward+backward/forward+backward_No_Vul.pkl" # 无漏洞保存路径
    Vul_func_path = "./vulsim/forward+backward/forward+backward_Vul.pkl" # 有漏洞保存路径
    model_path = './vulsim/forward+backward/word2vec(forward+backward).model'          # word2vec模型保存路径

    token_dim = 128 # token的维度

    time_start = time.time()
    print("training word2vec model....")
    # sentence_list = splite_sentence(slices_path) # 对数据集进行分词
    model = Word2Vec(sentences=IterCorpus(No_func_path,Vul_func_path),vector_size=token_dim,alpha=0.01,window=5,min_count=0,
                     sample=0.001,seed=1,workers=1,min_alpha=0.0001,sg=1,hs=0,negative=10) # 训练词向量,以每一行为单位，语料库为一个二维数组
    model.save(model_path)  # 保存模型
    # model = Word2Vec.load(model_path)  # 导入模型
    # print(model.wv['static'])
    time_end = time.time()
    sum_time = (time_end - time_start)/60

    print("模型训练完成！共花费时间：{} min".format(sum_time))


if __name__ == '__main__':
    main()