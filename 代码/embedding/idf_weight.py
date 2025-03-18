import pickle
import math
import tqdm
import torch
import torch.nn as nn
from word2vec import tokenize_code_line
from gensim.models import Word2Vec
import numpy as np

def gene_node_vec(no_funcs_info, vul_funcs_info, model, word_weight, node_num, vector_size, embedding_path, type_node, one_hot, linear):
    FUNS_info = {} # 保存所有函数信息

    for funcs_info in [no_funcs_info,vul_funcs_info]:
        for func in tqdm.tqdm(funcs_info):
            new_func_info = {} # 保持单个函数信息
            func_vec = [] # 切片每个结点的特征向量

            for node in funcs_info[func]['code_tokens']:
                node_vec = [] # 每个结点的tokens特征向量
                
                # 节点属性
                attr = node.split(',')[0].split('.')[0]
                if attr:
                    attr_vec = one_hot[type_node.index(attr)] # 属性编码
                else:
                    attr_vec = torch.zeros(vector_size) # 零向量替代

                # 节点
                tokens = tokenize_code_line(node.strip("\""))  # 分词
                for token in tokens:
                    token_vec = model.wv[token]
                    node_vec.append(token_vec)  # 不加权

                feature = torch.from_numpy(np.sum(node_vec,axis=0))  # 对token的向量求和得到句子（结点）向量,转化为tensor
                
                # 属性编码+节点语句向量
                # print(torch.cat((attr_vec, feature), dim=0).size())
                node_vec = linear(torch.cat((attr_vec, feature), dim=0)) #

                func_vec.append(node_vec.detach().numpy())  # 收集结点向量，得到整个切片的向量
            
            # 统一切片的结点个数
            if len(func_vec) < node_num:  # 结点太少，补零
                # func_vec += [np.zeros(vector_size*2,dtype=float).tolist()]*(node_num-len(func_vec))
                func_vec += [np.zeros(128,dtype=float).tolist()]*(node_num-len(func_vec))
            if len(func_vec) > node_num:   # 结点太多，截断
                func_vec = func_vec[:node_num]

            # new_func_info['code_tokens'] = funcs_info[func]['code_tokens']    # 切片序列信息
            new_func_info['tokens_vector'] = func_vec                       # 每个切片的结点向量特征
            new_func_info['CDG_adj'] = funcs_info[func]['CDG_adj']  # CDG
            new_func_info['DDG_adj'] = funcs_info[func]['DDG_adj']  # DDG
            new_func_info['PDG_adj'] = funcs_info[func]['PDG_adj']  # PDG
            new_func_info['label'] = funcs_info[func]['label'] # 当前函数的标签

            FUNS_info[func]=new_func_info

    # 保存结点特征
    with open(embedding_path, 'wb') as f:
        pickle.dump(FUNS_info, f)
    
    print("切片结点特征已保存...")

def merge_corpus(No_funcs_info, Vul_funcs_info, weight_path):
    """
    统计语料库，统计包含每个词的文档数,输出每次词的IDF(改)权重
    """
    corpus = []        # 语料库
    vocab = {}         # 统计包含每个词的文档数
    word_weight = {}   # 每个词的权重
    type_node = []

    print("正在获取语料库...")
    # 生成语料库
    for type_ in [No_funcs_info, Vul_funcs_info]:
        for func in tqdm.tqdm(type_):
            func_doc = [] # 单个切片
            for line in type_[func]['code_tokens']:

                if line.split(',')[0].split('.')[0] not in type_node:
                    type_node.append(line.split(',')[0].split('.')[0])

                tokens = tokenize_code_line(line.strip("\""))  # 分词
                for token in tokens:
                    func_doc.append(token)

            corpus.append(func_doc)

    # print("正在统计包含每个词的文档数...")
    # # 得到包含每个词的文档数
    # num_docs = len(corpus) # 文档个数
    # for sentence in tqdm.tqdm(corpus):
    #     words = set(sentence)  # 数组变为集合，做了一次去重
    #     for word in words:
    #         vocab[word] = vocab.get(word, 0) + 1

    # print("正在计算每个词的权重...")
    # # 改进IDF算法，获取每个词的权重
    # for word in vocab.keys():
    #     weight = math.log((num_docs / vocab.get(word, 0))+math.e-1) # 改进IDF值
    #     word_weight[word] = weight
    
    # # 保存词的权重
    # with open(weight_path, 'wb') as f:
    #     pickle.dump(word_weight, f)
    # print("词的权重已保存!")

    return word_weight,type_node

def gene_ont_hot(type_node):
    # 定义元素列表
    num_elements = len(type_node)
    embedding_dim = num_elements  # 设置嵌入维度为元素的数量，实现 one-hot 效果

    # 创建 Embedding 层
    embedding = nn.Embedding(num_embeddings=num_elements, embedding_dim=embedding_dim)
    # 手动设置 Embedding 层的权重，使其为 one-hot 向量形式
    one_hot_weights = torch.eye(num_elements)
    embedding.weight.data = one_hot_weights

    # 将元素映射为索引
    element_indices = torch.tensor([type_node.index(elem) for elem in type_node], dtype=torch.long)

    # 获取嵌入向量
    embeddings = embedding(element_indices)

    return embeddings 

def main():
    No_func_path = "./vulsim/backward/backward_No_Vul.pkl" # 无漏洞保存路径
    Vul_func_path = "./vulsim/backward/backward_Vul.pkl" # 有漏洞保存路径
    weight_path = './vulweight/word_weight.pkl'        # 每个词的权重保存路径
    model_path = './vulweight/word2vec(backward).model'          # word2vec模型保存路径
    embedding_path = '../vulweight/dataset/func_embedding.pkl'   # 节点特征和结构特征

    node_num = 100      # 切片最大结点数

    # 无漏洞信息
    with open(No_func_path,'rb') as f:
        No_funcs_info = pickle.load(f)
    # 有漏洞信息
    with open(Vul_func_path,'rb') as f:
        Vul_funcs_info = pickle.load(f)
    
    # 有漏洞信息
    with open(weight_path,'rb') as f:
        word_weight = pickle.load(f)

    # 得到每个词权重
    word_weight, type_node= merge_corpus(No_funcs_info, Vul_funcs_info, weight_path)
    vector_size = len(type_node)  # 结点向量维度

    one_hot = gene_ont_hot(type_node) # 生成one-hot编码
    linear = nn.Linear(in_features=128+len(type_node),out_features=128)
    
    model = Word2Vec.load(model_path)  # 导入模型
    
    print("正在计算切片图的结点向量...")
    # 获取PDG图结点的特征
    gene_node_vec(No_funcs_info,Vul_funcs_info, model, word_weight, node_num, vector_size, embedding_path, type_node, one_hot, linear) # 加权

if __name__ == '__main__':
    main()