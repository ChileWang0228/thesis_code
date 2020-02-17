#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.2
@Created on 2019-1-27 19:56
@Author:ChileWang
@algorithm：
将生成的csv文件的列名一致性
并构造训练集和测试集
"""
import pandas as pd
import numpy as np
# pd.set_option('display.max_columns', None)  # 显示所有列
csv_file_name = ['2010.csv', '2012.csv', '2014.csv', '2016.csv']
json_file_name = ['2010.json', '2012.json', '2014.json', '2016.json']
# 因为队员疏忽导致某些列不一致，需要删除以下列来保持列名一致性
drop_columns_2010 = ['qa201acode', 'wa1age', 'we101', 'wl1', 'fa4', 'tb2_a_p']
drop_columns_2012 = ['qa302ccode', 'we101', 'wl1', 'tb1b_a_f', 'fa4', 'cfps2010_gender', 'tb2_a_p']
drop_columns_2014 = ['ear201a', 'we101', 'wl1', 'code_b_1', 'tb2_a_p']
drop_columns_2016 = ['kz202_b_1', 'tb2_a_p']
drop_columns_list = [drop_columns_2010, drop_columns_2012, drop_columns_2014, drop_columns_2016]

consistent_file_dir = 'data_consistent_output_file/'  # 一致性csv文件存放地址
data_to_train_and_test = 'data_to_train_and_test/'  # 训练集和测试集csv文件存放地址


def build_consistent_csv():
    """
    生成列名一致的CSV文件
    :return:
    """
    # 取2010年的列名当做所有表格新的特证名
    new_columns_name = pd.read_csv('read_json_output_file/' + csv_file_name[0]).\
        drop(drop_columns_list[0], axis=1).columns.values.tolist()
    for i in range(len(csv_file_name)):
        pd_data = pd.read_csv('read_json_output_file/' + csv_file_name[i])
        new_pd_data = pd_data.drop(drop_columns_list[i], axis=1)  # 丢弃多余的列
        # 生成新的一致性csv文件
        columns_name = new_pd_data.columns.values.tolist()  # 要被修改的列名
        new_col = dict(zip(columns_name, new_columns_name))  # 压缩成字典
        new_pd_data.rename(columns=new_col, inplace=True)  # 改名
        new_pd_data.to_csv(consistent_file_dir + 'new_' + csv_file_name[i])


def build_train_and_test_set():
    """
    构造训练集和测试集
    取10~14年的数据做训练集：
        标签:比如取2014年fid与2012年fid的交集，存在的交集说明fid还在,fid中的pid设置标签为0：仍然属于该fid，并未分家；
        否则标签设置成1。
    最后取16年的数据集做测试集合
    :return:生成训练集与测试集的CSV文件
    """
    data_2010 = pd.read_csv(consistent_file_dir + 'new_' + csv_file_name[0])
    data_2012 = pd.read_csv(consistent_file_dir + 'new_' + csv_file_name[1])
    data_2014 = pd.read_csv(consistent_file_dir + 'new_' + csv_file_name[2])
    data_2016 = pd.read_csv(consistent_file_dir + 'new_' + csv_file_name[3])
    # columns = data_2010.columns.values.tolist()  # 列名
    data_year_list = [data_2010, data_2012, data_2014, data_2016]
    # 由于数据合并后出现大量的nan, 暂时不用含有大量nan的数据，因此要将数据分成两大块
    new_data_list_model1 = []  # 模型１的数据　　含有少量nan的数据
    new_data_list_model2 = []  # 模型2的数据　　　含有大量nan的数据
    for data_year in data_year_list:
        data_year.dropna(subset=['pid'], inplace=True)  # 删掉pid列为空的行
        row_na_index = list(np.where(np.isnan(data_year['urban']))[0])  # 找出urban id为空的行,以此为分割线将两份数据分开
        new_data_list_model1.append(data_year[0: row_na_index[0]].copy())
        new_data_list_model2.append(data_year[row_na_index[0]:].copy())

    # 给数据上标签
    new_data_2010 = new_data_list_model1[0]
    new_data_2012 = new_data_list_model1[1]
    new_data_2014 = new_data_list_model1[2]
    new_data_2016 = new_data_list_model1[3]
    # 若2012年的fid在交集中，　则说明家庭并未流失，pid仍然属于原先的fid,label赋值为0
    # 若不在其中，则说明家庭流失，pid已经不属于原先的fid，label赋值为１
    common_fid = set(new_data_2010['fid']) & set(new_data_2012['fid'])
    new_data_2010['label'] = new_data_2010.fid.apply(lambda x: 0 if x in common_fid else 1)
    common_fid = set(new_data_2012['fid']) & set(new_data_2014['fid'])
    new_data_2012['label'] = new_data_2012.fid.apply(lambda x: 0 if x in common_fid else 1)
    common_fid = set(new_data_2014['fid']) & set(new_data_2016['fid'])
    new_data_2014['label'] = new_data_2014.fid.apply(lambda x: 0 if x in common_fid else 1)

    # 构造训练集
    new_data_2010.to_csv(data_to_train_and_test + 'train_set.csv', index=None)  # 不要索引
    # 在csv文件中追加
    new_data_2012.to_csv(data_to_train_and_test + 'train_set.csv', index=None, mode='a+', header=False)  # 不要索引
    new_data_2014.to_csv(data_to_train_and_test + 'train_set.csv', index=None, mode='a+', header=False)  # 不要索引

    # 构造测试集
    test_set = new_data_2016.copy()
    test_set.to_csv(data_to_train_and_test + 'test_set.csv', index=False)


if __name__ == '__main__':
    build_consistent_csv()  # 生成一致性CSV文件
    build_train_and_test_set()  # 生成训练集和测试集
    # a = pd.DataFrame({'A': ['null', 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    # a.columns = [1, 2, 3]
    # a[1] = a[1].replace('null', np.nan)  # 可以用数字做列，　replace函数是右边替换左边
    # print(a[1])
