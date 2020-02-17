#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.1
测试平台：Windows
@Created on 2019/4/2 9:31
@Author:Cheng
@Algorithm：
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


TIME_THRESHOLD = 3 # 急加速急减速时间阈值
RAPIDLY_DECELERATION_THRESHOLD = -5 # 急减速加速度阈值
def add_rapidly_deceleration_label(df):
    """
    添加急减速标签rap_dec，计算急减速时间rap_dec_time
    :param df: DataFrame对象
    :return: DataFrame对象
    """
    # 默认没有急减速
    df['rap_dec_time'] = 0
    df['rap_dec'] = 0
    # 子路程不可能急减速的情况
    if len(df) <= 1:
        return df
    # 取出速度和时间戳这两列，复制移位一列
    df_select = df[['timestamp', 'gps_speed']]
    df_select_copy = df_select.shift(-1).copy()
    # 速度用上一最后时刻的速度填充，时间用最后的时间+1
    df_select_copy.loc[len(df_select_copy) - 1] = [df_select.loc[len(df_select) - 1, 'timestamp'] + 1, df_select.loc[len(df_select) - 1, 'gps_speed']]
    df_out = df_select_copy - df_select
    df[['speed_diff', 'time_diff']] = df_out[['gps_speed', 'timestamp']]
    need_index = np.where(df['speed_diff'] <= RAPIDLY_DECELERATION_THRESHOLD)
    need_index = need_index[0]
    for i in range(len(need_index)):
        time_diff = df.loc[need_index[i], 'time_diff']
        speed_diff = df.loc[need_index[i], 'speed_diff']
        if (time_diff <= TIME_THRESHOLD) and ((speed_diff * 1000) / (3600 * time_diff)) <= RAPIDLY_DECELERATION_THRESHOLD:
            df.loc[need_index[i] + 1, 'rap_dec'] = 1
            df.loc[need_index[i] + 1, 'rap_dec_time'] = time_diff
    return df


URGENT_ACCELERATION_THRESHOLD = 5 # 急加速加速度阈值
def add_urgent_acceleration_label(df):
    """
    添加急加速标签urg_acc，计算急减速时间urg_acc_time
    :param df: DataFrame对象
    :return: DataFrame对象
    """
    # 默认没有急加速
    df['urg_acc_time'] = 0
    df['urg_acc'] = 0
    # 子路程不可能急加速的情况
    if len(df) <= 1:
        return df
    # 取出速度和时间戳这两列，复制移位一列
    df_select = df[['timestamp', 'gps_speed']]
    df_select_copy = df_select.shift(-1).copy()
    # 速度用上一最后时刻的速度填充，时间用最后的时间+1
    df_select_copy.loc[len(df_select_copy) - 1] = [df_select.loc[len(df_select) - 1, 'timestamp'] + 1, df_select.loc[len(df_select) - 1, 'gps_speed']]
    df_out = df_select_copy - df_select
    df[['speed_diff', 'time_diff']] = df_out[['gps_speed', 'timestamp']]
    need_index = np.where(df['speed_diff'] >= URGENT_ACCELERATION_THRESHOLD)
    need_index = need_index[0]
    for i in range(len(need_index)):
        # time_diff = df.loc[need_index[i] + 1, 'timestamp'] - df.loc[need_index[i], 'timestamp']
        time_diff = df.loc[need_index[i], 'time_diff']
        speed_diff = df.loc[need_index[i], 'speed_diff']
        if (time_diff <= TIME_THRESHOLD) and ((speed_diff * 1000) / (3600 * time_diff)) >= URGENT_ACCELERATION_THRESHOLD:
            df.loc[need_index[i] + 1, 'urg_acc'] = 1
            df.loc[need_index[i] + 1, 'urg_acc_time'] = time_diff
    return df


def sort_date(date):
    """
    用来对df中的日期排序
    :param date: date是08-01这样的字符串
    :return: 0801这样的整型
    """
    str1 = date.split('-')[0]
    str2 = date.split('-')[1]
    ret = int(str1 + str2)
    return ret


def add_date(date):
    # 只取出日期
    date = date.split(' ')[0]
    date = date[5:]
    if "/" in date:
        date1 = date.split('/')[0]
        date2 = date.split('/')[1]
    else:
        date1 = date.split('-')[0]
        date2 = date.split('-')[1]
    if len(date1) == 1:
        date1 = '0' + date.split('-')[0]
    if len(date2) == 1:
        date2 = '0' + date.split('-')[1]
    date = date1 + '-' + date2
    return date


if __name__ == "__main__":
    out_dataset_1_path = "F:\Szu\out_data_1\\"

    upload_car_list = ['AA00002', 'AB00006', 'AD00003', 'AD00013', 'AD00053', 'AD00083', 'AD00419', 'AF00098',
                       'AF00131', 'AF00373']

    columns = ['vehicle', 'date', 'avg_speed(km/h)', 'driving_distance(km)', 'urgent_acc', 'rapidly_dec']

    out_dir = r"F:\Szu\Question_one\\"
    out_table = pd.DataFrame(columns=columns)

    count = 0
    for car in upload_car_list:
        date_path = out_dataset_1_path + car + "\\"
        date_dirs = os.listdir(date_path)

        temp_df = pd.read_csv(r"F:\Szu\szu\\" + car + ".csv")
        # avg_speed = temp_df['gps_speed'].mean()
        temp_df['date'] = temp_df.location_time.apply(lambda x: add_date(x))

        for date_dir in date_dirs:

            # 判断是否是文件夹
            if not (os.path.isfile(date_path + date_dir)):
                date_file_list = os.listdir(date_path + date_dir)  # 各日期子文件夹下的csv文件
                urgent_acc_count = []  # urg_acc
                # urgent_acc_time = []  # urg_acc_time
                rapidly_dec_count = []  # rap_dec
                # rapidly_dec_time = []  # rap_dec_time

                date = date_dir.split('_')[-1]  # 文件的日期
                # 日期有些是8-4，有些是08-04，需处理成统一的08-04这种形式
                date1 = date.split('-')[0]
                date2 = date.split('-')[1]
                if len(date1) == 1:
                    date1 = '0' + date.split('-')[0]
                if len(date2) == 1:
                    date2 = '0' + date.split('-')[1]
                date = date1 + '-' + date2

                mileage = temp_df[temp_df['date'] == date]['mileage'].values
                mileage = mileage[-1] - mileage[0]
                driving_distance = mileage

                avg_speed = temp_df[temp_df['date'] == date]['gps_speed'].mean()

                for date_file in date_file_list:
                    print(date_file)

                    df = pd.read_csv(date_path + date_dir + "\\" + date_file)
                    df = add_rapidly_deceleration_label(df)
                    df = add_urgent_acceleration_label(df)

                    urgent_acc_count.append(df['urg_acc'].sum())
                    # urgent_acc_time.append(df['urg_acc_time'].sum())

                    rapidly_dec_count.append(df['rap_dec'].sum())
                    # rapidly_dec_time.append(df['rap_dec_time'].sum())

                out_table.loc[count] = [car, date, avg_speed, driving_distance, np.array(urgent_acc_count).sum(), np.array(rapidly_dec_count).sum()]
                count += 1


            if if_dir_exists(out_dir):
                # 排序
                out_table = out_table.sort_values(by='vehicle')
                out_table.to_csv(out_dir + "Question_one.csv", index=None)