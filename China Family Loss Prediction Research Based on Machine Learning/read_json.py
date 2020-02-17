#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.2
@Created on 2019-1-18 09:47
@Author:ChileWang
@algorithm：
读取组员挑选出来的重要特征的json文件，并从stata文件中抽出形成CSV文件
"""
import json
import pandas as pd
from pandas.io.stata import StataReader

adult_dir_path = 'JSON/adult_%s_important_dict.json'  # 存放成人重要特征的json文件路径
child_dir_path = 'JSON/child_%s_important_dict.json'  # 存放小孩重要特征的json文件路径
famcof_dir_path = 'JSON/famcof_%s_important_dict.json'  # 存放家庭社区重要特征的json文件路径
family_dir_path = 'JSON/family_%s_important_dict.json'  # 存放家庭重要特征的json文件路径
years = ['2010', '2012', '2014', '2016']
json_data_dir = [adult_dir_path, child_dir_path, famcof_dir_path, family_dir_path]  # 存放json文件的路径
data_year_dir = '/home/chilewang/Desktop/Data/'  # 存放数据的路径


def write_json(important_feature_dict, file_name):
    """
    将字典写入json文件
    :param important_feature_dict:
    :param file_name:
    :return:
    """
    important_feature_json = json.dumps(important_feature_dict)
    with open('read_json_output_file/%s.json' % file_name, 'w', encoding='UTF-8') as fw:
        fw.write(important_feature_json)


def read_json():
    """
    读取json文件
    :return:放回各个stata文件的存放路径
    """

    data_path = []
    for i in range(4):
        temp = []
        data_year_dict = dict()
        for j in range(len(json_data_dir)):
            json_path = json_data_dir[j] % years[i]  # 构建读取json数据的路径
            with open(json_path, 'r', encoding='UTF-8') as fr:
                important_feature_dict = dict(json.load(fr))
            # 将每一年的所有表的特征形成字典
            for key in important_feature_dict.keys():
                if key not in data_year_dict.keys():
                    data_year_dict[key] = important_feature_dict[key]

            stata_file_name = important_feature_dict['file_name']  # 读取其对应的stata文件名
            stata_data_path = data_year_dir + years[i] + '/' + stata_file_name  # 构建要读取的stata文件的路径
            important_feature = list(important_feature_dict.keys())  # 该表除去文件名的所有重要特征
            del important_feature[0]
            temp_dict = dict()
            temp_dict[stata_data_path] = important_feature
            temp.append(temp_dict)  # 存放当年所有表格的路径和重要特征的字典
        data_path.append(temp)
        write_json(data_year_dict, years[i])  # 将特征写入json文件
    return data_path


def data_merge(df_list, file_name):
    """
    合并DataFrame
    :param df_list:
    :return: 并生成CSV文件
    """
    columns0 = set(df_list[0].columns.values.tolist())  # 特征名转换为集合
    columns1 = set(df_list[1].columns.values.tolist())  # 特征名转换为集合
    on_columns = list(columns0 & columns1)  # 基于二者的交集合并
    pd_important_feature_merge = pd.merge(df_list[0], df_list[1], on=on_columns, how='outer')  # outer链接保留所有项
    for i in range(2, len(df_list)):
        columns = set(pd_important_feature_merge.columns.values.tolist())  # 特征名转换为集合
        columns_i = set(df_list[i].columns.values.tolist())  # 特征名转换为集合
        on_columns = list(columns & columns_i)  # 基于二者的交集合并
        pd_important_feature_merge = pd.merge(pd_important_feature_merge, df_list[i], on=on_columns, how='outer')

    pd_important_feature_merge.to_csv(file_name, index=False)  # 不要索引


def convert_to_df():
    """
    将stata文件中的重要特征抽取出来，每一年合成一张dataframe表格
    :return:
    """
    # 装载每个年份合并之后的DataFrame的文件名
    data_merge_file_name = ['read_json_output_file/2010.csv', 'read_json_output_file/2012.csv',
                            'read_json_output_file/2014.csv', 'read_json_output_file/2016.csv']

    data_path = read_json()  # 读取stata文件存放地址
    for i in range(len(data_path)):
        temp = []
        for j in range(len(data_path[i])):
            for key in data_path[i][j].keys():
                stata_data_path = key  # 当年某表的存放路径
                columns_name = data_path[i][j][key]  # 该表对应的重要特征
                print(columns_name)
                stata_data = StataReader(stata_data_path, convert_categoricals=False)  # 读取stata文件
                pd_important_feature = pd.DataFrame(stata_data.read())[columns_name]  # 将格式转成DataFrame,并读取其重要特征
                temp.append(pd_important_feature)
        data_merge(temp, data_merge_file_name[i])  # 合并并生成csv文件
        print('-------------------------')


if __name__ == '__main__':
    convert_to_df()



