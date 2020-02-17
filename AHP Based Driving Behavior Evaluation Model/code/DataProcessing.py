#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.1
测试平台：Windows
@Created on 2019/4/1 22:21
@Author:Cheng
@Algorithm：
"""
import pandas as pd
import numpy as np
import os
import time

def if_dir_exists(dir):
    """
    # 判断目录是否存在，不存在则创建
    :param dir: 目录
    :return: True
    """
    if os.path.exists(dir):
        return True
    else:
        os.makedirs(dir)
        return True


def extract_sub_trace(file_path, out_path):
    """
    提取子路程
    从最原始的文件中拆分出子路程
    每个子路程放到以原文件命名的文件夹中
    :param file_path: 原始文件所在目录，格式为"F:\Szu\szu\\"，最后要两个斜杠
    :param out_path: 输出文件目录，格式为"F:\Szu\szu_filter\\"，最后要两个斜杠
    :return:
    """
    file_nameList = os.listdir(file_path)
    for file_name in file_nameList:
        get = pd.read_csv(file_path + file_name)
        print("abstract sub trace from: " + file_name)
        # 去掉文件名的后缀
        file_name = file_name.split('.')[0]
        # 以array的形式返回速度所在列的值
        # file = get.iloc[:, -2].values
        file = get['gps_speed'].values
        # file = file.values
        try:
            """
            有些文件可能速度全都为0，所以筛选出速度大于0的路程后索引会出错
            """
            file = np.where(file > 0)
            i = 0
            j = 0
            labels = []
            inn = []
            start = file[0][0]
        except IndexError:
            # 如果车的没有子路程，创建一个空目录，并进入下一个循环
            os.makedirs(out_path + file_name)
            continue
        for x in range(len(file[0])):
            y = file[0][x]
            if start != y:
                i = i + 1
                j = 0
                start = y
                labels.append(inn)
                inn = []
            start = start + 1
            inn.append(y)
            j += 1
            if x == (len(file[0]) - 1):
                labels.append(inn)

        for i in range(len(labels)):
            save = get.iloc[labels[i][0]: labels[i][-1] + 1]
            if if_dir_exists(out_path + file_name):
                save.to_csv(out_path + file_name + "\\" + file_name + "_" + str(i + 1) + ".csv")


def add_avg_speed_to_src_file(src_path, sub_path):
    """
    为源文件添加平均速度
    :param src_path: 源文件目录，F:\Szu\szu\\，后面两斜杠
    :param sub_path: 子路程所在目录，F:\Szu\szu_filter\\，后面两斜杠
    :return: DataFrame对象
    """
    sub_dir_names = os.listdir(sub_path)
    for sub_dir_name in sub_dir_names:
        fileList = os.listdir(sub_path + sub_dir_name)
        source_file = src_path + sub_dir_name
        src_df = pd.read_csv(source_file + ".csv")
        src_df['avg_speed'] = 0
        for file in fileList:
            df = pd.read_csv(sub_path + sub_dir_name + "\\" + file)
            src_df.loc[df.iloc[:, 0], 'avg_speed'] = df['gps_speed'].mean()
        # if if_dir_exists(out_path):
        #     print("add avg speed to: " + sub_dir_name)
        #     out_name = sub_dir_name + ".csv"
        #     src_df.to_csv(out_path + out_name)
        return src_df


def add_timestamp_to_sub_trace(sub_abs_dir_file):
    """
    为子路程（或其他文件）添加时间戳
    增加一列，存放对应的时间戳
    进行这一步需要先提取出所有子路程
    :param sub_abs_dir_file: 所有子路程文件所在位置绝对路径，格式为"F:\Szu\szu_filter\AA00001\AA00001_1.csv"
    :return: DataFrame对象
    """
    df = pd.read_csv(sub_abs_dir_file)
    df['timestamp'] = 0
    # 转成时间戳，先判断时间是否有秒，即时间形式是否是8:11、23:11这样的
    if (3 < (len(df.loc[0, 'location_time'].split(" ")[1])) < 6):
        df['timestamp'] = df.location_time.apply(lambda x: time.mktime(time.strptime(x, '%Y/%m/%d %H:%M')) if "/" in x else time.mktime(time.strptime(x, '%Y-%m-%d %H:%M')))
        try:
            """
            有些子路程只有1行数据，通过两行判断的方式会出错
            """
            if (df.loc[0, 'location_time'] != df.loc[1, 'location_time']):
                # 1、2行不同，说明第一行是分钟的最后一秒，即8:11，8:12
                df.loc[0, 'timestamp'] = df.loc[0, 'timestamp'] + 60
            else:  # 1、2行相同，但前面几行可能不是1个分钟内全部的数据
                df['timestamp'] = df['timestamp'] + [x for x in range(len(df['timestamp']))]
                # try: # 如果子路程没有数据，会出错
                unique_time = df['location_time'].unique()[0] # 取出一开始的时间
                length = len(df[df['location_time'] == unique_time])
                df.loc[0: length - 1, 'timestamp'] = df.loc[0: length - 1, 'timestamp'] + list(range(60 - length + 1, 60 + 1))
                # except KeyError:
                #     return df
        except KeyError:
            """
            出现KeyError说明只有1行数据
            """
            if (3 < (len(df.loc[0, 'location_time'].split(" ")[1])) < 6):
                df['timestamp'] = df.location_time.apply(lambda x: time.mktime(time.strptime(x, '%Y/%m/%d %H:%M')) if "/" in x else time.mktime(time.strptime(x, '%Y-%m-%d %H:%M')))
            else: # 带秒
                df.drop_duplicates(['location_time'], keep='last', inplace=True) # 去重
                df['timestamp'] = df.location_time.apply(lambda x: time.mktime(time.strptime(x, '%Y/%m/%d %H:%M:%S')) if "/" in x else time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    # 如果时间形式不是8:11的，那就是8:11:11这样带秒的
    else:
        df.drop_duplicates(['location_time'], keep='last', inplace=True)  # 去重
        df['timestamp'] = df.location_time.apply(lambda x: time.mktime(time.strptime(x, '%Y/%m/%d %H:%M:%S')) if "/" in x else time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    df.index = range(len(df))
    return df


def get_time(time):
    j = 0
    begin = 0
    end = 0
    for i,s in enumerate(time):
        if (s == '/' or s =='-') and j == 0:
            j = j + 1
            begin = i + 1
        if (s == '/' or s == '-') and j == 1:
            j = j + 1
        if s == ' ' and j == 2:
            end = i
            j = 0
            break
    return time[begin:end].replace('/','-')


def split_src_by_date(in_path, out_path, action=1):
    """
    将源文件拆分成日期和子路程
    :param file_add_timestamp_dir: 源文件地址（先添加了时间戳和平均速度）
    :param out_path: 输出路径
    :return:
    """
    f = os.listdir(in_path)
    # f = os.listdir("/home/chilewang/Documents/szu_timestamp")
    # f = ['AA00004.csv']
    for j in f:
        file = pd.read_csv(in_path+j)
        time =[]
        for i in range(len(file['location_time'])):
            time.append(get_time(str(file['location_time'][i])))
        file['time'] = time
        name = j.split('.')[0]
        # os.makedirs('/home/chilewang/Documents/out_dataset_0/'+name)
        os.makedirs(out_path + name)
        begin_n=0
        end_n=0
        bol_s=''
        for n,s in enumerate(time):
            if n==0:
                bol_s = s
                continue
            if bol_s != s:
                end_n = n
                savefile = file.iloc[begin_n:end_n]
                savefile.to_csv(out_path + name + '/' + name + '_' + bol_s + '.csv')
                bol_s = s
                begin_n = end_n
            if n==len(time)-1:
                savefile = file.iloc[begin_n:-1]
                savefile.to_csv(out_path + name + '/' + name + '_' + bol_s + '.csv')

    cars = os.listdir(out_path)
    for car in cars:
        days = os.listdir(out_path + car)
        for day in days:
            get= pd.read_csv(out_path + car + "/" + day)
            if get['gps_speed'][0] != 0:
                print("跨越"+day)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            #将速度大于０的连续的全部行提取出来，并且分为多个csv
            # file=get.iloc[:,-4]######修改
            file = get['gps_speed']
            file=file.values
            if action == 1:
                file =np.where(file > 0)
            else:
                file = np.where(file == 0)
            i=0
            j=0
            labels = []
            inn=[]
            try:
                start = file[0][0]
            except:
                continue
            for x in range(len(file[0])):
                y = file[0][x]
                if start != y :
                    i =i+1
                    j=0
                    start = y
                    labels.append(inn)
                    inn=[]
                start = start + 1
                inn.append(y)
                # labels[i][j]=y
                j =j+1
                if x==(len(file[0])-1):
                    labels.append(inn)
            os.makedirs(out_path + car + '/' + day.split('.')[0])
            for i in range(len(labels)):
                save=get.iloc[labels[i][0]:labels[i][-1]+1]
                save.to_csv(out_path + car + '/' + day.split('.')[0] + '/' + str(i+1) + '__' + day)

            print("finish" + day)


if __name__ == "__main__":
    src_file_path = r"F:\Szu\szu\\" # 源文件路径
    src_file_list = os.listdir(src_file_path)
    for file in src_file_list:
        # 为源文件添加时间戳
        print(file)
        df = add_timestamp_to_sub_trace(src_file_path + file)
        out_dir = r"F:\Szu\szu_timestamp\\"
        if if_dir_exists(out_dir):
            df.to_csv(out_dir + file)
    # 提取有速度的子路程
    if if_dir_exists(r"F:\Szu\szu_timestamp_subtrace\\"):
        extract_sub_trace(r"F:\Szu\szu_timestamp\\", r"F:\Szu\szu_timestamp_subtrace\\")
    # 按日期提取有速度的子路程
    if if_dir_exists(r"F:\Szu\out_data_1\\"):
        split_src_by_date(r"F:\Szu\szu_timestamp\\", r"F:\Szu\out_data_1\\", 1)
    # 按日期提取没速度的子路程
    if if_dir_exists(r"F:\Szu\out_data_0\\"):
        split_src_by_date(r"F:\Szu\szu_timestamp\\", r"F:\Szu\out_data_0\\", 0)

