import os
# import pickle
# import difflib
import pandas as pd

# def label(info_vul, info_novul ,modified_lines, outfile):
#     start_point = 0
#     diff = difflib.unified_diff(info_vul.split('\n'), info_novul.split('\n')) # 对比原函数和被修复后的函数之间的差异性
#     diff_list = [i for i in diff if i != '']

#     for i,line in enumerate(diff_list):
#         if line.startswith("@@"):
#             start_point=i
#             break

#     for code_line in diff_list[start_point:]:  # 记录原函数中被删除的代码
#         if code_line.startswith("-"):
#             modified_lines.setdefault(outfile,[]).append(code_line.strip('-'))
    
def process_raw_data(filter_csv_path, raw_csv_path):
    if os.path.exists(filter_csv_path):  # 原始数据已经被处理，直接读取处理后得到的文件；否则，重新处理
        pd_filter = pd.read_csv(filter_csv_path)  # 读取处理后得到的文件
    else:
        pd_raw = pd.read_csv(raw_csv_path)  # 读取原始文件
        filter_csv = pd_raw.loc[pd_raw["vul"] == 1]  # 以行为单位，选择表中有漏洞的数据
        filter_csv.to_csv(filter_csv_path, index=False)  # 单独保存表中有漏洞的数据
        pd_filter = pd.read_csv(filter_csv_path)  # 读取处理后得到的文件
    return pd_filter

def main():
    filter_csv_path = './dataset/filtered_data.csv' # csv文件中，vul栏为1的漏洞数据保存地址
    raw_csv_path = './dataset/MSR_data_cleaned.csv' # 原始数据集地址
    output_path = './dataset/Vul_dataset'           # 输出漏洞文件的地址

    # pkl_path = './dataset/modified_lines.pkl'  # 函数被修改后的信息保存路径
    # modified_lines = {} # 键为文件名，值为被修改的代码行

    print("正在csv文件中提取漏洞文件...")
    pd_filter = process_raw_data(filter_csv_path, raw_csv_path)  # 从数据集中，提取出漏洞函数所在的行数据
    file_list = []  # 保存输出函数的保存路径
    cnt_1 = 0  # 统计保存函数的个数

    for index, row in pd_filter.iterrows():  # 对每一行数据分别处理
        print("\r文件处理进度:{}/10900".format(index+1),end="")

        project_name = row["project"]  # 获取漏洞函数所在项目名称
        hash_vaule = row['commit_id']  # 获取漏洞函数commit_id
        file_name = project_name + "_" + hash_vaule  # 以所属项目名称和commit_id的hash作为保存漏洞文件的文件名
        outfile = output_path + '/' + file_name  # 漏洞函数保存路径

        file_name_cnt = 0
        outfile_new = outfile
        while outfile_new in file_list:  # 处理漏洞函数保存路径重复的情况
            outfile_new = outfile + '_' + str(file_name_cnt)
            file_name_cnt += 1
        file_list.append(outfile_new)

        if not os.path.exists(outfile_new):  # 文件夹不存在，则创建
            os.makedirs(outfile_new)

        func_before = row['func_before']  # 未修复漏洞函数
        func_after = row["func_after"]    # 已修复漏洞函数
        vul_file_name = '1_'+ file_name + '.c'
        novul_file_name = '0_' + file_name+ '.c'

        # 将未修复漏洞和已修复漏洞分别保存
        with open(outfile_new + '/'+ vul_file_name, 'w', encoding='utf-8') as f_vul:
            f_vul.write(func_before)
            cnt_1 += 1
        with open(outfile_new + '/' + novul_file_name, 'w', encoding='utf-8') as f_novul:
            f_novul.write(func_after)
            cnt_1 += 1
        
    #     label(func_before,func_after,modified_lines,outfile)
        
    # with open(pkl_path, 'wb') as f:  # 保存有漏洞函数和无漏洞函数之间的差异
    #     pickle.dump(modified_lines, f)

    print("\nfinish!一个生成了{}个c文件".format(cnt_1))

if __name__ == '__main__':
    main()