#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.1
测试平台：Windows
@Created on 2019/4/16 14:57
@Author:Cheng
@Algorithm：
输出各车评价均分
"""
import pandas as pd
import numpy as np
import os

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


def get_score(vehicle, df):
    """
    打分
    :param vehicle: 车名
    :param df: pandas.DataFrame对象
    :return: list，[安全分数，节能分数]
    """
    weights_security = [[0.08724146],
                        [0.17881193],
                        [0.22931698],
                        [0.08724146],
                        [0.17881193],
                        [0.23857624]]  # 安全驾驶打分权重
    weights_green = [[0.06997414],
                     [0.22953027],
                     [0.12071752],
                     [0.22953027],
                     [0.12071752],
                     [0.22953027]]  # 节能打分权重

    row = df[df['vehicle']==vehicle]
    val_security = row[['speed_std', 'urgent_acc', 'rapidly_dec', 'overspeed', 'fatigue_drive', 'coasting']].values
    val_green = row[['speed_std', 'long_idle', 'urgent_acc', 'rapidly_dec', 'overspeed', 'pre_heating']].values

    return [np.dot(val_security, weights_security)[0], np.dot(val_green, weights_green)[0, 0]]


def get_total_score(row):
    """
    打总分
    :param row: 一行数据
    :return: 总分
    """
    security = row['score_security']
    green = row['score_green']
    return 0.5 * security + 0.5 * green

if __name__ == "__main__":
    columns = ['vehicle', 'speed_std', 'urgent_acc', 'rapidly_dec', 'overspeed', 'fatigue_drive', 'coasting',
               'long_idle', 'pre_heating', 'score_security', 'score_green']
    # 默认100分
    out_table = pd.DataFrame(np.ones((450, len(columns))) * 100, columns=columns)
    file_path = r"F:\Szu\final1\\"
    file_list = os.listdir(file_path)
    count = 0
    for file in file_list:
        print(file)
        vehicle = file.split('.')[0]  # 车牌
        out_data_list = [vehicle]  # 按顺序存放columns的内容
        df_temp = pd.read_csv(file_path + file)
        df_mean = df_temp.mean()  # 求均值
        try:
            out_data_list.append(df_mean['std_score'])  # 车速稳定性分数
            out_data_list.append(df_mean['urgent_acc_score'])  # 急加速分数
            out_data_list.append(df_mean['rapidly_dec_score'])  # 急减速分数
            out_data_list.append(df_mean['overspeed_score'])  # 超速分数
            out_data_list.append(df_mean['fatigue_score'])  # 疲劳驾驶分数
            out_data_list.append(df_mean['coasting_score'])  # 熄火滑行分数
            df_temp_2 = pd.read_csv(r"F:\Szu\final2\\" + file)
            df_mean_2 = df_temp_2.mean()
            out_data_list.append(df_mean_2['long_idle_score'])  # 超长怠速分数
            out_data_list.append(df_mean_2['pre_heating_score'])  # 怠速预热分数

            out_table.loc[count, columns[: -2]] = out_data_list  # 写入数据
        except Exception:
            out_table.loc[count, 'vehicle'] = vehicle  # 没有数据只车牌

        # 打分和总分
        out_table.loc[count, ['score_security', 'score_green']] = get_score(vehicle, out_table)

        count += 1
    out_path = r"F:\Szu\Question_three\\"

    out_table['total_score'] = out_table.apply(lambda row: get_total_score(row), axis=1)

    if if_dir_exists(out_path):
        out_table = out_table.fillna(100).copy()
        out_table.to_csv(out_path + "Question_three.csv")