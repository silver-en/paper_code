import os
import torch
import tqdm
import pickle
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from tensorboardX import SummaryWriter
from torch_geometric.data import Data
from prettytable import PrettyTable
from torch_geometric.loader import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from  torch_geometric.nn.conv import SAGEConv,GCNConv,GatedGraphConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")  # 忽视警告

# 设置随机种子
seed = 71926                       # 生成随机数种子的初值
np.random.seed(seed)               # 为numpy随机数生成器设置随机数种子
torch.manual_seed(seed)            # 为cpu的随机数生成器设置随机数种子
torch.cuda.manual_seed(seed)       # 为当前cuda的随机数生成器设置随机数种子
torch.cuda.manual_seed_all(seed)   # 为所有cuda的随机数生成器设置随机数种子

# 定义多头自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        # x的形状是[batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.size()

        # 线性变换
        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)    # [batch_size, seq_len, embed_dim]
        V = self.value(x)  # [batch_size, seq_len, embed_dim]

        # 计算注意力权重
        energy = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # [batch_size, seq_len, seq_len]
        attention = F.softmax(energy, dim=-1)  # [batch_size, seq_len, seq_len]

        # 应用注意力权重
        output = torch.matmul(attention, V)  # [batch_size, seq_len, embed_dim]

        # 选择最重要的一个64维数据
        # 这里我们取每个样本的加权和作为最终输出
        final_output = output.sum(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]

        return final_output


class Vulsim(nn.Module):
    def __init__(self, batch, node_num, input_dim,output_dim, dropout):
        super(Vulsim,self).__init__()
        self.batch = batch # batch个数
        self.node_num = node_num  # 结点个数
        self.input_dim = input_dim

        ## 图神经网络
        # self.sage = SAGEConv(in_channels=input_dim,out_channels=input_dim,aggr='max',normalize=True)  # 源模型
        # self.gcn = GCNConv(in_channels=input_dim,out_channels=input_dim)  # 图神经网络换为GCN
        self.ggnn = GatedGraphConv(out_channels=input_dim,num_layers=1,aggr='max')  # 图神经网络换为两层的GGNN

        ## 序列
        # self.gru = nn.GRU(input_size = input_dim,hidden_size = input_dim,batch_first=True,dropout=dropout,num_layers=3)  # GRU
        self.blstm = nn.LSTM(input_size = input_dim,hidden_size = input_dim,batch_first=True,dropout=0.5, bidirectional=True,num_layers=2)
        
        self.conv_1 = nn.Sequential(
            # conv_output:<batch,卷积核个数（输出通道数），结点个数-kernel_size+1>  
            nn.Conv1d(in_channels = output_dim*3,  # 输入通道等于结点特征维度
                      out_channels = output_dim,  # 输出通道数等于卷积核的个数
                      kernel_size = 3) ,          # 卷积核的kernel_Size，卷积核尺寸（输入通道，kernel_size）
            # nn.BatchNorm1d(num_features=output_dim),  # 正则化
            nn.ReLU()  # 激活函数
        )
        self.conv_2 = nn.Sequential(
            # conv_output:<batch,卷积核个数（输出通道数），结点个数-kernel_size+1>  
            nn.Conv1d(in_channels = output_dim,  # 输入通道等于结点特征维度
                      out_channels = 32,  # 输出通道数等于卷积核的个数
                      kernel_size = 3) ,          # 卷积核的kernel_Size，卷积核尺寸（输入通道，kernel_size）
            # nn.BatchNorm1d(num_features=64),  # 正则化
            nn.ReLU() # 激活函数
            # nn.Dropout(0.1)
        )

        self.pool_1 = nn.Sequential(
            nn.AdaptiveMaxPool1d(output_size=1),  # 最大化池化
            # nn.Dropout(dropout),
            nn.Flatten(1)
        )

        self.pool_2 = nn.MaxPool1d(kernel_size=8)

        # 创建自注意力模块实例，使用3个头
        self.attention_1 = SelfAttention(embed_dim=32)
        self.attention_2 = SelfAttention(embed_dim=output_dim*2)

        self.dropout = nn.Dropout(dropout) # Dropout层
        # self.softmax = nn.Softmax(dim=1)  # softmax层
        self.sigmoid = nn.Sigmoid()  # softmax层

        self.mlp = nn.Sequential(
            nn.Linear(in_features=node_num, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2)
        )
        # self.linear = nn.Linear(in_features=64,out_features=2)
        self.linear = nn.Linear(in_features=32,out_features=2)
        self.linear1 = nn.Linear(in_features=64*3,out_features=2)

    def forward(self, cdg_f, cdg_a, ddg_f, ddg_a, pdg_f, pdg_a):
        # out_ggnn = self.ggnn(feature,topo)  # SAGE聚合结点信息
        # output_1 = out_ggnn.reshape(self.batch,self.node_num,-1) # 经过图神经网络处理后的数据转换成批数据
        # class_1 = self.dropout(self.conv(output_1.permute(0,2,1)))
        # pool_1 = self.pool(class_1)

        # # 融合后卷积池化得到嵌入
        # concat = torch.cat((cdg, ddg, pdg),dim=-1)  # 拼接原特征矩阵和聚合后的特征矩阵
        # class_1 = self.dropout(self.conv_1(concat.permute(0,2,1)))
        # pool_1 = self.pool(class_1)
        # result = self.linear2(pool_1)

        # ## 图特征
        # # cdg
        # out_cdg = self.ggnn(cdg_f,cdg_a)  # GGNN聚合结点信息
        # cdg = out_cdg.reshape(self.batch,self.node_num,-1) # 经过图神经网络处理后的数据转换成批数据

        # # ddg
        # out_ddg = self.ggnn(ddg_f,ddg_a)  # GGNN聚合结点信息
        # ddg = out_ddg.reshape(self.batch,self.node_num,-1) # 经过图神经网络处理后的数据转换成批数据

        # # # pdg
        # out_pdg = self.ggnn(pdg_f,pdg_a)  # GGNN聚合结点信息
        # pdg = out_pdg.reshape(self.batch,self.node_num,-1) # 经过图神经网络处理后的数据转换成批数据
        
        # # cdg
        # class_cdg = self.dropout(self.conv_2(cdg.permute(0,2,1)))
        # # class_cdg = self.dropout(self.conv_2(cdg.permute(0,2,1)))
        # pool_cdg = self.pool_1(class_cdg)

        # # ddg
        # class_ddg = self.dropout(self.conv_2(ddg.permute(0,2,1)))
        # # class_ddg = self.dropout(self.conv_2(ddg.permute(0,2,1)))
        # pool_ddg = self.pool_1(class_ddg)

        # # # pdg
        # class_pdg = self.dropout(self.conv_2(pdg.permute(0,2,1)))
        # # class_pdg = self.dropout(self.conv_2(pdg.permute(0,2,1)))
        # pool_pdg = self.pool_1(class_pdg)


        # # 应用自注意力机制
        # concat = torch.stack([pool_cdg,pool_ddg,pool_pdg],dim=1) # 3个(32,64)-》（32，3,64）
        # output_1 = torch.squeeze(self.attention_1(concat),dim=1)
        # pool_out_1 = self.pool_1(self.conv_2(output_1.permute(0,2,1))) # 卷积、池化

        
        ## 序列特征
        feature = cdg_f.reshape(self.batch,self.node_num,-1)
        out,hn = self.blstm(torch.tensor(feature))
        output_2 = torch.squeeze(self.attention_2(out),dim=1)
        # res = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        pool_out_2 = self.pool_2(output_2)

        # 融合
        # concat = torch.concat((pool_out_2,output_1),dim=-1) # 拼接特征矩阵
        # print(output_1.size())
        # print(pool_out_2.size())
        # print(concat.size())
        # result = self.linear(concat)
        result = self.linear(pool_out_2)

        # concat = torch.concat((pool_cdg,pool_ddg,pool_pdg),dim=-1) # 拼接特征矩阵,(1,64)->(1,64*3)
        # print(concat.size())

        # concat = torch.stack([pool_cdg,pool_pdg],dim=1) # 3个(32,64)-》（32，3,64）
        # output = self.attention(concat)
        # result = self.linear(torch.squeeze(output,dim=1))

        # # 不用自注意力机制
        # concat = torch.concat((pool_cdg,pool_ddg,pool_pdg),dim=-1) # 拼接特征矩阵
        # result = self.linear1(concat)

        # result = self.linear(pool_pdg)

        # print(output.shape)  # 输出形状应该是[batch, 3, 64]

        # 如果需要将结果压缩为[batch, 1, 64]，可以取平均或使用其他聚合方法
        # final_output = output.mean(dim=1, keepdim=True)  # 取平均
        # print(final_output.shape)  # 输出形状应该是[1, 1, 64]
        

        # # class_1 = self.dropout(self.classify_1(output_1.permute(0,2,1)))
        # pool_1 = self.pool(class_1)
        # # result = self.linear1(pool_1)

        # # # 特征融合
        # concat = torch.concat((pool_1,pool_2),dim=-1) # 拼接特征矩阵
        # result = self.linear2(concat)

        return self.sigmoid(result)

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class ThreeGraphDataset:
    def __init__(self, data):
        self.data = data
        self.data_list = []
        self._generate_data()

    def _generate_data(self):
        for func in self.data:
            token_vector = np.array(func['tokens_vector'])  # 结点向量
            CDG_adj = func['CDG_adj']  # CDG
            DDG_adj = func['DDG_adj']  # DDG
            PDG_adj = func['PDG_adj']  # PDG
            label = func['label'] # 标签

            x = torch.tensor(token_vector,dtype=torch.float)  # 结点特征
            y = torch.tensor(int(label))  # 标签

            # CDG图
            x = torch.tensor(token_vector,dtype=torch.float)  # 结点特征
            coo_cdg = sparse.coo_matrix(CDG_adj)  # 密集矩阵转化为稀疏矩阵
            edge_cdg = torch.tensor(np.array([coo_cdg.row,coo_cdg.col]),dtype=torch.long)  # 拓扑结构
            cdg = Data(x=x,edge_index=edge_cdg)  # 构建数据

            # ddg图
            coo_ddg = sparse.coo_matrix(DDG_adj)  # 密集矩阵转化为稀疏矩阵
            edge_ddg = torch.tensor(np.array([coo_ddg.row,coo_ddg.col]),dtype=torch.long)  # 拓扑结构
            ddg = Data(x=x,edge_index=edge_ddg)  # 构建数据

            # pdg图
            coo_pdg = sparse.coo_matrix(PDG_adj)  # 密集矩阵转化为稀疏矩阵
            edge_pdg = torch.tensor(np.array([coo_pdg.row,coo_pdg.col]),dtype=torch.long)  # 拓扑结构
            pdg = Data(x=x,edge_index=edge_pdg)  # 构建数据

            # 将三个图存储在一个数据对象中
            data = Data()
            data.cdg = cdg
            data.ddg = ddg
            data.pdg = pdg
            data.y = torch.tensor(label, dtype=torch.long)  # 存储标签
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def collect_data(data):  
    # 创建图专属数据集
    pair_data_list = []
    for func in data:
        token_vector = func['tokens_vector']  # 结点向量
        CDG_adj = func['CDG_adj']  # CDG
        DDG_adj = func['DDG_adj']  # DDG
        PDG_adj = func['PDG_adj']  # PDG
        label = func['label'] # 标签

        x = torch.tensor(token_vector,dtype=torch.float)  # 结点特征
        y = torch.tensor(int(label))  # 标签

        # CDG图
        x = torch.tensor(token_vector,dtype=torch.float)  # 结点特征
        coo_cdg = sparse.coo_matrix(CDG_adj)  # 密集矩阵转化为稀疏矩阵
        edge_cdg = torch.tensor(np.array([coo_cdg.row,coo_cdg.col]),dtype=torch.long)  # 拓扑结构
        # ast = Data(x=x,edge_index=edge_index,y=y)  # 构建数据

        # ddg图
        coo_ddg = sparse.coo_matrix(DDG_adj)  # 密集矩阵转化为稀疏矩阵
        edge_ddg = torch.tensor(np.array([coo_ddg.row,coo_ddg.col]),dtype=torch.long)  # 拓扑结构
        # cfg = Data(x=x,edge_index=edge_index,y=y)  # 构建数据

        # pdg图
        coo_pdg = sparse.coo_matrix(PDG_adj)  # 密集矩阵转化为稀疏矩阵
        edge_pdg = torch.tensor(np.array([coo_pdg.row,coo_pdg.col]),dtype=torch.long)  # 拓扑结构
        # cfg = Data(x=x,edge_index=edge_index,y=y)  # 构建数据

        pair_data = PairData(x_cdg=x,edge_cdg=edge_cdg,
                            x_ddg=x,edge_ddg=edge_ddg,
                            x_pdg=x,edge_pdg=edge_pdg,
                            y_cdg=y,y_ddg=y,y_pdg=y
                            )

        pair_data_list.append(pair_data)

    return pair_data_list


class Vulsim_detector():
    def __init__(self, node_num, input_dim, output_dim, epochs, batch_size, \
                 learning_rate, decay, dropout):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.node_num = node_num      # 切片的节点数量
        self.input_dim = input_dim    # 切片结点特征向量的维度
        self.output_dim = output_dim  # 经过图神经网络聚集后的输出向量维度
        self.epochs = epochs      
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay = decay
        self.dropout = dropout
        self.best_f1 = 0
        self.write = SummaryWriter('./log_data/tokens_feature/')
 
        save_path = './model_data/tokens_feature/Vulweight.pkl'
        if not os.path.exists(os.path.dirname(save_path)): os.mkdir(os.path.dirname(save_path))
        self.save_path = save_path

    def preparation(self, train_data, test_data):
        # create datasets
        # self.train_set = collect_data(train_data)
        # self.test_set = collect_data(test_data)

        self.train_set = ThreeGraphDataset(train_data)
        self.test_set = ThreeGraphDataset(test_data)

        # create data loaders
        self.train_loader = DataLoader(self.train_set,batch_size=self.batch_size,shuffle=False,drop_last=True) 
        # print(len(self.train_loader))
        self.test_loader = DataLoader(self.test_set,batch_size=self.batch_size,shuffle=False,drop_last=True) 

        # helpers initialization
        self.model = Vulsim(self.batch_size, self.node_num,self.input_dim,self.output_dim,self.dropout)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate,weight_decay=self.decay)  # 设置优化器
        # self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=50,
        #     num_training_steps=len(self.train_loader) * self.epochs
        # )
        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def train(self):

        train_table = PrettyTable(['typ', 'epo', 'loss', 'ACC'])
        test_table = PrettyTable(['typ', 'epo', 'loss', 'ACC', 'pre', 're', 'f1'])
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            train_loss, train_acc = self.fit()  # 训练
            self.write.add_scalar("train_loss", train_loss, epoch+1)  # 写到tensorboard里去
            self.write.add_scalar("train_acc", train_acc, epoch+1)
            train_table.add_row(["tra", str(epoch+1), format(train_loss, '.4f'), format(train_acc*100, '.4f')] )
            print(train_table)

            test_loss, test_acc, test_pre, test_re, test_f1 = self.eval()  # 验证（测试）
            self.write.add_scalar("valid_loss", test_loss, epoch+1)
            self.write.add_scalar("valid_acc", test_acc, epoch+1)
            self.write.add_scalar("valid_pre", test_pre, epoch+1)
            self.write.add_scalar("valid_re", test_re, epoch+1)
            self.write.add_scalar("valid_f1", test_f1, epoch+1)
            test_table.add_row(["val", str(epoch+1), format(test_loss, '.4f'), format(test_acc*100, '.4f'),\
                                format(test_pre*100, '.4f'), format(test_re*100, '.4f'),format(test_f1*100, '.4f')])
            print(test_table)
            
            if epoch == 0:
                best_loss, best_acc, best_pre, best_re = test_loss, test_acc, test_pre, test_re

            if test_f1 > self.best_f1:
                self.best_f1 = test_f1
                best_loss, best_acc, best_pre, best_re = test_loss, test_acc, test_pre, test_re
                # print("best model:Val Loss: {:.2} | Acc: {:.2f}% | P: {:.2f}% | R: {:.2f}% | f1: {:.2f}%".format(
                #     best_loss, best_acc*100, best_pre*100, best_re*100, self.best_f1*100))
                torch.save(self.model.state_dict(), self.save_path)  # 保存当前网络状态

            print("best model:Val Loss: {:.2} | Acc: {:.2f}% | P: {:.2f}% | R: {:.2f}% | f1: {:.2f}%".format(
                    best_loss, best_acc*100, best_pre*100, best_re*100, self.best_f1*100))
    
    def fit(self):
        self.model = self.model.train() # 启用batch normalization和dropout
        train_loss = 0
        train_acc = 0
        
        progress_bar = tqdm.tqdm(self.train_loader)
        for data in progress_bar: # 以batch为单位取数据
            data_cdg = DataLoader(data.cdg,batch_size=self.batch_size)
            for cdg_f in data_cdg:
                cdg = cdg_f

            data_ddg = DataLoader(data.ddg,batch_size=self.batch_size)
            for ddg_f in data_ddg:
                ddg = ddg_f

            data_pdg = DataLoader(data.pdg,batch_size=self.batch_size)
            for pdg_f in data_pdg:
                pdg = pdg_f
            # cdg图
            cdg.x = cdg.x.to(self.device)
            cdg.edge_index = cdg.edge_index.to(self.device)

            # ddg图
            ddg.x = ddg.x.to(self.device)
            ddg.edge_index = ddg.edge_index.to(self.device)

            # pdg图
            pdg.x = pdg.x.to(self.device)
            pdg.edge_index = pdg.edge_index.to(self.device)

            # forward + backward + optimize
            pre_outputs = self.model(cdg.x, cdg.edge_index, ddg.x, ddg.edge_index, pdg.x, pdg.edge_index)  # 前向传播
            loss = self.loss_fn(pre_outputs.cpu(),data.y)  # 求loss

            # zero the parameter gradients
            self.optimizer.zero_grad()
            loss.backward()  # 反向传播求梯度
            self.optimizer.step() # 更新所有参数

            predict_labels = torch.argmax(pre_outputs, dim=1).cpu()  # 返回每一行最大的元素的下标，作为模型预测结果

            # 使用sklearn中的评价函数的时候，参数需要在cpu中
            acc = accuracy_score(data.y.cpu(), predict_labels)  # 计算每一次训练的准确度
            progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {acc:.3f}')

            train_loss += loss.item()  # 对每一此训练的loss进行求和
            train_acc += acc

        return train_loss / len(self.train_loader), train_acc / len(self.train_loader)  # 返回一轮epoch的平均损失和准确率

    def eval(self):
        print("start evaluating...")

        self.model = self.model.eval()   # 不启用 batch normalization 和 dropout
        valid_loss = 0
        valid_acc = 0
        valid_pre = 0
        valid_recall = 0
        valid_f1 = 0
        progress_bar = tqdm.tqdm(self.test_loader)

        with torch.no_grad():
            for data in progress_bar:
                data_cdg = DataLoader(data.cdg,batch_size=self.batch_size)
                for cdg_f in data_cdg:
                    cdg = cdg_f

                data_ddg = DataLoader(data.ddg,batch_size=self.batch_size)
                for ddg_f in data_ddg:
                    ddg = ddg_f

                data_pdg = DataLoader(data.pdg,batch_size=self.batch_size)
                for pdg_f in data_pdg:
                    pdg = pdg_f

                # cdg图
                cdg.x = cdg.x.to(self.device)
                cdg.edge_index = cdg.edge_index.to(self.device)

                # ddg图
                ddg.x = ddg.x.to(self.device)
                ddg.edge_index = ddg.edge_index.to(self.device)

                # pdg图
                pdg.x = pdg.x.to(self.device)
                pdg.edge_index = pdg.edge_index.to(self.device)

                pre_outputs = self.model(cdg.x, cdg.edge_index, ddg.x, ddg.edge_index, pdg.x, pdg.edge_index)  # 前向传播
                loss = self.loss_fn(pre_outputs.cpu(),data.y)           # 求loss

                predict_labels = torch.argmax(pre_outputs, dim=1).cpu()  # 返回每一行最大的元素的下标，作为模型预测结果
                acc = accuracy_score(data.y.cpu(), predict_labels)    # 计算准确度
                pre = precision_score(data.y.cpu(),predict_labels)    # 计算精度
                recall = recall_score(data.y.cpu(), predict_labels)   # 计算召回率
                f1 = f1_score(data.y.cpu(), predict_labels)  # 计算F1值

                valid_loss += loss.item()  # 对每一此训练的loss进行求和
                valid_acc += acc
                valid_pre += pre
                valid_recall += recall
                valid_f1 += f1

                progress_bar.set_description(
                f'loss: {loss.item():.3f}, acc : {acc:.3f}')

        return valid_loss / len(self.test_loader), valid_acc / len(self.test_loader), \
               valid_pre / len(self.test_loader), valid_recall / len(self.test_loader), valid_f1 / len(self.test_loader)
