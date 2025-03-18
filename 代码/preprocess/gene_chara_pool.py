import os
import re
import glob
import tqdm
import numpy as np
import pickle
from multiprocessing import Pool  # 进程池
import networkx as nx
from functools import partial

def get_api_node(codes_dict):
    with open("./sensitive_func.pkl", "rb") as f: # 读取文件中保存的所有危险函数
        sensi_func = pickle.load(f)
    new_sensi_func = []  # 保存处理后的危险函数
    for func in sensi_func:
        new_sensi_func.append(func.replace('_','/_').replace('.','/.').replace('*','/*'))  # 替换掉危险函数中的特殊字符

    node_api_list=[]  # 用于保存调用危险函数的结点
    for func in new_sensi_func:  # 找出调用危险函数的结点
        for key, code in codes_dict.items():
            if re.findall(func,code): # 查找代码行中是否包含危险函数
                node_api_list.append(key) # 保存危险结点编号及代码
    
    return node_api_list

def get_pointer_node(codes_dict):
    codes_dict = nx.get_node_attributes(G,'label')  # 获取每个结点的编号和对应的代码
    node_point_list=[]  # 用于保存包含指针的结点

    for key, value in codes_dict.items():  # 处理原始代码
        type = value[2:value.index(",")]  # 获取节点类型
        code = value[value.index(",") + 1:-2].strip()  # 获取结点源代码
        if type != 'UNKNOWN':
            if code.find("*VAR") != -1:   # 找出包含数组的结点
                    node_point_list.append(key)

    return node_point_list

def get_array_node(codes_dict):

    # 返回与数组有关的结点及其对应编号
    node_array_list=[]  # 与数组有关结点
    for key, value in codes_dict.items():  # 处理原始代码
        type = value[2:value.index(",")]  # 获取节点类型
        code = value[value.index(",") + 1:-2].strip()  # 获取结点源代码
        if type != 'UNKNOWN':
            if code.find("[") != -1:   # 找出包含数组的结点
                node_array_list.append(key)

    return node_array_list

def get_all_expre(codes_dict):
    node_expre_list=[]  # 用于保存包含表达式的结点

    for key,value in codes_dict.items():
        results = None
        type = value[2:value.index(",")]  # 获取节点类型
        code = value[value.index(",") + 1:-2].strip()  # 获取结点源代码
        
        if type != 'UNKNOWN':
            if code.find("=") != -1:
                new_code = code.split('=')[-1].strip() # 获取等号右边的内容
                pattern = re.compile("((_|[A-Za-z]|[0-9]+)\w*(\s)*(\+|\-|\*|\/)(\s)*(_|[A-Za-z]|[0-9]+)\w*)")
                results = re.search(pattern, new_code)  # 等号右边内容是否有表达式
                if results != None:
                    node_expre_list.append(key)

    return node_expre_list

def slice_node(G,sensi_list):  # 切片算法
    nodes_list = [] # 与切片兴趣点有关的节点
    
    for point in sensi_list:
        back_nodes = nx.dfs_successors(G,source=point) # 后向切片
    
        prodromal = [i for node in back_nodes.values() for i in node]   # 前驱结点
        prodromal.append(point)  # 补充切片准则

        for node in prodromal:
            if node not in nodes_list:
                nodes_list.append(node)

    return nodes_list

def coll_edges(G,sorted_nodes):
    # 收集与漏洞有关的CDG、DDG边
    edges_dict = nx.get_edge_attributes(G,'label')  # 获取每个结点的编号和对应的代码

    CDG_edge = [] # CDG边
    DDG_edge = [] # DDG边
    for nums,value in edges_dict.items():
        # 只要与漏洞有关的节点的边
        if (nums[0] in sorted_nodes) and (nums[1] in sorted_nodes): 
            edge_kind= value.split(':')[0].split('"')[1] # 真实的标签
            if edge_kind == "CDG":
                CDG_edge.append((nums[0],nums[1]))
            if edge_kind == "DDG":
                DDG_edge.append((nums[0],nums[1]))
    
    return CDG_edge,DDG_edge

def sub_graph(code_num, edge, node_number):
    # 构建子图
    G = nx.DiGraph()
    for node in code_num:
        G.add_node(node)
    # 添加结点和有向边
    for nodes in edge:
        G.add_edge(nodes[0],nodes[1])
    # AST结构信息
    adj = nx.adjacency_matrix(G,nodelist=code_num).todense()  # 按照原PDG的顺序生成邻接矩阵

    # 统一拓扑结构的维度
    if len(adj) < node_number:  # 结点太少，补零
        new_topo = np.pad(adj,((0,node_number-len(adj)),((0,node_number-len(adj)))))
    elif len(adj) > node_number:   # 结点太多，截断
        new_topo = adj[:node_number,:node_number]
    else:
        new_topo = adj

    return new_topo

def PDG_adj(CDG_edge,DDG_edge,sorted_nodes,node_number):
    PDG_edge = [] # pdg边
    for edge_1 in CDG_edge:
        PDG_edge.append(edge_1)
    for edge_2 in DDG_edge:
        if edge_2 not in PDG_edge:
            PDG_edge.append(edge_2)

    PDG_adj = sub_graph(sorted_nodes, PDG_edge, node_number) # DDG子图
    return PDG_adj

def func_info():
    func_info = {} # 收集单个函数的所有信息

    dot_file = os.path.join(dot_path, dir, file,'1-pdg.dot')  # 函数主要dot文件
    if os.path.exists(dot_file):
        G = nx.MultiDiGraph(nx.nx_pydot.read_dot(dot_file))  # 读取dot文件，并构建有向图
        codes_dict = nx.get_node_attributes(G,'label')  # 获取每个结点的编号和对应的代码

        # 获取切片兴趣点，并去重
        node_api_list = get_api_node(codes_dict) # 获取与API有关的结点
        node_array_list = get_array_node(codes_dict) # 获取与数组有关的结点
        node_pointer_list = get_pointer_node(codes_dict) # 获取与指针有关的结点
        node_expr_list = get_all_expre(codes_dict) 
        sensi_list = list(set(sum([node_api_list, node_pointer_list, node_array_list, node_expr_list],[])))  # 汇总所有的危险结点,并去重

        # 切片，获取与切片有关的节点编号
        num_list = list(codes_dict.keys())  # dot文件中的结点编号
        G_reverse = nx.reverse(G)  # 反转箭头

        backward = slice_node(G_reverse,sensi_list) # 后向切片有关节点
        forward = slice_node(G,sensi_list) # 前向切片有关节点
        wards = [] # 所有切片节点
        for no_1 in backward:
            wards.append(no_1)
        for no_2 in forward:
            if no_2 not in wards:
                wards.append(no_2)

        if wards: # 如果切片没有节点，就用原图
            sorted_nodes = sorted(wards, key=num_list.index)  # 按照dot文件中的编号进行排序
        else:
            sorted_nodes = num_list
        all_node.append(len(sorted_nodes))

        ## 收集节点token序列、各个图的邻接矩阵
        CDG_edge,DDG_edge = coll_edges(G,sorted_nodes) # 收集与漏洞有关的CDG和DDG边
        CDG_adj = sub_graph(sorted_nodes, CDG_edge, node_number) # CDG邻接矩阵
        DDG_adj = sub_graph(sorted_nodes, DDG_edge, node_number) # DDG邻接矩阵
        pdg_adj = PDG_adj(CDG_edge,DDG_edge,sorted_nodes,node_number)   # PDG邻接矩阵

        # 获取节点序列
        code_tokens = []
        for node in sorted_nodes:
            code_tokens.append(codes_dict[node][2:-2])
        
        func_info['code_tokens'] = code_tokens[:node_number] # 切片序列信息
        func_info['CDG_adj'] = CDG_adj # CDG拓扑结构信息
        func_info['DDG_adj'] = DDG_adj # DDG拓扑结构信息
        func_info['PDG_adj'] = pdg_adj # PDG拓扑结构信息
        func_info['label'] = label # 当前函数的标签


    return file,func_info


if __name__ == '__main__':
    dot_path = "./dataset/dot_data"    # dot文件路径
    info_path = "./dataset/ffm(forward).pkl" # 文件保存路径
    all_node_path = "./dataset/all_node_num(forward).pkl" # 统计切片节点个数

    node_number = 200 # 节点个数
    all_node = [] # 收集所有节点个数
    FUN_info = {} # 保存所有数据的信息

    pool_num = 20
    pool = Pool(pool_num) # 创建进程池

    if type == 'parse':
        print("正在解析c文件...")
        file_dir = glob.glob(input+'/*')  # 获取所有C文件所在的文件夹
        pool.map(partial(joern_parse,joern_path=joern_path,out_bin_dir=out_bin_path),get_all(file_dir))
        pool.close() # 关闭进程池，使其不再接受新的请求
        pool.join()  # 等待进程池中的所有子进程执行完毕，再执行接下来的代码
  
    for dir in ["No_Vul","Vul"]:
        if dir == 'Vul':
            label = 1
        elif dir == 'No_Vul':
            label = 0

        files = glob.glob(dot_path+'/'+dir+'/*') # 某个类型所有函数

        counter = multiprocessing.Value('i', 0)  # 整数类型的共享变量

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # 将 counter 作为额外参数传递给 process_file
            results = pool.starmap(process_file, [(path, counter) for path in file_paths])

        for file in tqdm.tqdm(os.listdir(dot_path+'/'+dir)):
            func_info = {} # 收集单个函数的所有信息

            dot_file = os.path.join(dot_path, dir, file,'1-pdg.dot')  # 函数主要dot文件
            if os.path.exists(dot_file):
                G = nx.MultiDiGraph(nx.nx_pydot.read_dot(dot_file))  # 读取dot文件，并构建有向图
                codes_dict = nx.get_node_attributes(G,'label')  # 获取每个结点的编号和对应的代码

                # 获取切片兴趣点，并去重
                node_api_list = get_api_node(codes_dict) # 获取与API有关的结点
                node_array_list = get_array_node(codes_dict) # 获取与数组有关的结点
                node_pointer_list = get_pointer_node(codes_dict) # 获取与指针有关的结点
                node_expr_list = get_all_expre(codes_dict) 
                sensi_list = list(set(sum([node_api_list, node_pointer_list, node_array_list, node_expr_list],[])))  # 汇总所有的危险结点,并去重

                # 切片，获取与切片有关的节点编号
                num_list = list(codes_dict.keys())  # dot文件中的结点编号
                G_reverse = nx.reverse(G)  # 反转箭头

                backward = slice_node(G_reverse,sensi_list) # 后向切片有关节点
                forward = slice_node(G,sensi_list) # 前向切片有关节点
                wards = [] # 所有切片节点
                for no_1 in backward:
                    wards.append(no_1)
                for no_2 in forward:
                    if no_2 not in wards:
                        wards.append(no_2)

                if wards: # 如果切片没有节点，就用原图
                    sorted_nodes = sorted(wards, key=num_list.index)  # 按照dot文件中的编号进行排序
                else:
                    sorted_nodes = num_list
                all_node.append(len(sorted_nodes))

                ## 收集节点token序列、各个图的邻接矩阵
                CDG_edge,DDG_edge = coll_edges(G,sorted_nodes) # 收集与漏洞有关的CDG和DDG边
                CDG_adj = sub_graph(sorted_nodes, CDG_edge, node_number) # CDG邻接矩阵
                DDG_adj = sub_graph(sorted_nodes, DDG_edge, node_number) # DDG邻接矩阵
                pdg_adj = PDG_adj(CDG_edge,DDG_edge,sorted_nodes,node_number)   # PDG邻接矩阵

                # 获取节点序列
                code_tokens = []
                for node in sorted_nodes:
                    code_tokens.append(codes_dict[node][2:-2])
                
                func_info['code_tokens'] = code_tokens[:node_number] # 切片序列信息
                func_info['CDG_adj'] = CDG_adj # CDG拓扑结构信息
                func_info['DDG_adj'] = DDG_adj # DDG拓扑结构信息
                func_info['PDG_adj'] = pdg_adj # PDG拓扑结构信息
                func_info['label'] = label # 当前函数的标签

                FUN_info[file] = func_info

    with open(info_path, 'wb') as f:
        pickle.dump(FUN_info, f)
    
    with open(all_node_path, 'wb') as f:
        pickle.dump(all_node, f)