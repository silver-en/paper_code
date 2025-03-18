VulSim

预处理：
1、raw_dat_preprocess.py 提取原始数据集中的漏洞文件，并且记录源文件（有漏洞的文件）中被删除的代码。输出：./preprocess/dataset/Vul_dataset/
2、normalization.py 对raw_dat_preprocess.py生成的文件进行删除注释、规范化、去重操作。对比修复前后的代码，找出被修改的代码行。输出：./preprocess/dataset/pred_dataset/
3. gen_sub_cfg.py 构建mfg图，并对切片兴趣点进行后向切片
4. gene_dot.py 对后向切片得到的代码片段进行解析，得到CPG图
5. gene_chara.py 得到四种子图（AST,CFG,CDG,DDG）和序列

向量化




VulFusion










3、gene_dot.py 生成预处理后的C文件的PDG图，以dot形式保存在./preprocess/dataset/dot_data/... 目录下
4、clean_dot.py 清理不合适的dot文件
5、gene_slices.py 生成各个函数的四种类型切片，输出目录：./preprocess/dataset/slice_data/... 目录下
6、colla_slice.py 对每个函数而言，去除重复的切片，并汇总所有切片代码及拓扑结构，保存在每个函数的all_slices目录下
                  对于无漏洞函数和有漏洞函数，去重重复切片，汇总所有信息，保存在./dataset/目录下
7、label.py 给每个切片打标签。

modified_lines.pkl 保存有漏洞的函数中被删除的代码行
API.txt：保存系统特有的函数名
sensitive_func.pkl 保存危险函数名
-----------------------------------

向量化：
1、word2vec.py 以结点为单位，将切片进行向量化

----------------------------------
模型训练：
1、collect_slice.py 将所有函数的切片、切片对应的拓扑结构和切片对应的标签汇总起来。保存在./dataset/raw_data.txt中
2、main.py  构建数据集-》切分数据集-》模型训练-》模型验证-》模型测试


预处理：
1、raw_dat_preprocess.py 提取原始数据集中的漏洞文件，并且记录源文件（有漏洞的文件）中被删除的代码。输出：./preprocess/dataset/Vul_dataset/
2、gene_dot.py 生成预处理后的C文件的PDG图，以dot形式保存在./preprocess/dataset/dot_data/... 目录下
3、normalization.py 对


数据集情况：(dot文件夹里的文件)
函数：有漏洞函数：4256个，无漏洞函数：4119个,总数：8375个
切片：有漏洞切片个数：20342个，无漏洞切片个数：49869个，总数：70211个



./dataset/dot_data.tar.gz 保存的是目前使用joern能解析的msr数据。



注：在进行函数间向量化的时候，可能会对一些函数没进行处理（_110）。