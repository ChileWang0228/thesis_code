#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.1
测试平台：Windows
@Created on 2019/5/2 10:29
@Author:Cheng
@Algorithm：
"""
import pandas as pd
import numpy as np
import time
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
        file = get.iloc[:, -2].values
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


def extract_by_date(in_path, out_path):
    """
    按日期提取
    :param in_path:
    :param out_path:
    :return:
    """
    df = pd.read_csv(in_path)
    df['date'] = df.location_time.apply(lambda x: x.split(' ')[0].replace('/', '-')[5:] if '/' in x else x.split(' ')[0][5:])
    unique_date = df['date'].unique()
    for date in unique_date:
        if if_dir_exists(out_path + date):
            df[df['date'] == date].to_csv(out_path + date + "\\" + date + '.csv')


def add_avg_speed_to_src_file(src_path, sub_path, out_path):
    """
    为源文件添加平均速度
    :param src_path: 源文件目录，F:\Szu\szu\\，后面两斜杠
    :param sub_path: 子路程所在目录，F:\Szu\szu_filter\\，后面两斜杠
    :param out_path: 输出目录，F:\Szu\szu_filter_speed\\，后面两斜杠
    :return:
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


# 增加超速label列，名称为over_speed，超速1，没有0
SPEED_THRESHOLD = 100 # 超速阈值，100km/h
# 添加是否超速标签over_speed，计算超速时间time_op
def judge_over_speed(df):
    """
    传入一个DataFrame对象，根据要求判断是否超速
    超速速度为100km/h（27.78m/s），且持续时长为3s或以上
    超速的最后一刻设over_speed为1，最后一刻索引对应的time_op填上超速的时间
    :param df: DataFrame对象
    :return: DataFrame对象
    """
    # 默认没有超速
    df['time_op'] = 0
    df['over_speed'] = 0
    # 子路程不可能超速的情况
    if len(df) <= 1:
        return df
    need_index = np.where(df['gps_speed'] > SPEED_THRESHOLD)
    need_index = need_index[0]
    begin = 0
    end = 0
    for i in range(len(need_index)):
        try:
            if need_index[i] + 1 == need_index[i + 1]:
                end += 1
            else:
                # df.loc[need_index[begin]: need_index[end], 'over_speed'] = 1
                df.loc[need_index[end], 'over_speed'] = 1
                df.loc[need_index[end], 'time_op'] = df.loc[need_index[end], 'timestamp'] - df.loc[need_index[begin], 'timestamp']
                begin = i
                end = i
        except IndexError:
            # df.loc[need_index[begin]: need_index[end], 'over_speed'] = 1
            # df.loc[need_index[end], 'over_speed'] = 1
            df.loc[need_index[end], 'time_op'] = df.loc[need_index[end], 'timestamp'] - df.loc[need_index[begin], 'timestamp']
            if (df.loc[need_index[end], 'timestamp'] - df.loc[need_index[begin], 'timestamp']) > 0:
                df.loc[need_index[end], 'over_speed'] = 1
    return df


TIME_THRESHOLD = 3 # 急加速急减速时间阈值
RAPIDLY_DECELERATION_THRESHOLD = -5 # 急减速加速度阈值
# 添加急减速标签rap_dec，计算急减速时间rap_dec_time
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
# 添加急加速标签urg_acc，计算急加速时间urg_acc_time
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


COASTING_WITH_ENGINE_OFF_THRESHOLD = 50 # 熄火滑行速度上限
# 添加熄火滑行标签coasting_with_engine_off，计算熄火滑行时间cweo_time
def add_coasting_with_engine_off(df):
    """
    添加熄火滑行标签coasting_with_engine_off，计算熄火滑行时间cweo_time
    熄火滑行：ACC为0，0<速度<50km/h（13.89m/s），持续时间大于等于3s
    :param df: DataFrame对象
    :return: DataFrame对象
    """
    # 默认没有熄火滑行
    df['cweo_time'] = 0
    df['coasting_with_engine_off'] = 0
    # 子路程不可能熄火滑行的情况
    if len(df) <= 1:
        return df
    need_index = np.where((0 < df['gps_speed']) & (df['gps_speed'] < COASTING_WITH_ENGINE_OFF_THRESHOLD) & (df['acc_state'] == 0))
    need_index = need_index[0]
    begin = 0
    end = 0
    for i in range(len(need_index)):
        try:
            if need_index[i] + 1 == need_index[i + 1]:
                end += 1
            else:
                # df.loc[need_index[begin]: need_index[end], 'over_speed'] = 1
                time_diff = df.loc[need_index[end], 'timestamp'] - df.loc[need_index[begin], 'timestamp']
                if time_diff >= 3:
                    df.loc[need_index[end], 'coasting_with_engine_off'] = 1
                    df.loc[need_index[end], 'cweo_time'] = time_diff
                begin = i
                end = i
        except IndexError:
            time_diff = df.loc[need_index[end], 'timestamp'] - df.loc[need_index[begin], 'timestamp']
            if time_diff >= 3:
                df.loc[need_index[end], 'coasting_with_engine_off'] = 1
                df.loc[need_index[end], 'cweo_time'] = time_diff
    return df


# 用速度为0的子路程做，添加超长怠速标签long_idle，计算超长怠速时间long_idle_time
# 还有怠速预热标签，怠速预热时间
def add_long_idle_label(df):
    """
    添加超长怠速标签long_idle，计算超长怠速时间long_idle_time
    acc = 1, v = 0
    T >= 60s
    :param df: DataFrame对象
    :return: DataFrame对象
    """
    # 默认没有超长怠速
    df['long_idle_time'] = 0
    df['long_idle'] = 0
    # 子路程不可能超长怠速的情况
    if len(df) <= 1:
        df['pre_heating'] = df['long_idle']
        df['pre_heating_time'] = df['long_idle_time']
        return df
    need_index = np.where(df['acc_state'] == 1)
    need_index = need_index[0]
    begin = 0
    end = 0
    for i in range(len(need_index)):
        try:
            if need_index[i] + 1 == need_index[i + 1]:
                end += 1
            else:
                # df.loc[need_index[begin]: need_index[end], 'over_speed'] = 1
                time_diff = df.loc[need_index[end], 'timestamp'] - df.loc[need_index[begin], 'timestamp']
                if time_diff >= 60:
                    df.loc[need_index[end], 'long_idle'] = 1
                    df.loc[need_index[end], 'long_idle_time'] = time_diff
                begin = i
                end = i
        except IndexError:
            time_diff = df.loc[need_index[end], 'timestamp'] - df.loc[need_index[begin], 'timestamp']
            if time_diff >= 60:
                df.loc[need_index[end], 'long_idle'] = 1
                df.loc[need_index[end], 'long_idle_time'] = time_diff
    # 添加怠速预热和超长怠速标签
    df['pre_heating'] = df['long_idle']
    df['pre_heating_time'] = df['long_idle_time']
    return df


def fatigue_driving(path):
    """
    输入文件所在路径
    :param path:
    :return:疲劳驾驶次数，疲劳驾驶时间，总驾驶时间
    """
    number = 0
    file = pd.read_csv(path+'.csv')
    speed_list = []
    zero_list = []
    indexs_list = []
    time = file['timestamp']
    speed = file['gps_speed']
    speed = speed.values
    start = speed[0]
    end = speed[-1]
    if start == 0:
        state = 0
    else:
        state = 1
    for index,v in enumerate(speed):
        if v == 0 and state == 1:
            indexs_list.append(index)
            state = 0
        if v != 0 and state == 0:
            indexs_list.append(index)
            state = 1
    # if start == 0:
    #     indexs_list_1 = indexs_list[1:]
    # if end == 0:
    #     indexs = indexs_list_1[:-1]
    j = 0
    for ind in range(len(indexs_list)):
        if ind == 0:
            if start == 0:
                continue
            if start != 0:
                speed_list.append(time[indexs_list[ind]] - time[0])
                j = 1
                continue
        if j == 0:
            speed_list.append(time[indexs_list[ind]] - time[indexs_list[ind-1]])
            j = 1
        else:
            zero_list.append(time[indexs_list[ind]] - time[indexs_list[ind-1]])
            j = 0

        if ind == len(indexs_list)-1:
            if end == 0:
                continue
            if end != 0:
                speed_list.append(time[len(time)-1] - time[indexs_list[ind]]+1)
                j = 1
    # speed_list = [5100,    100,    10000,     10000,      14500,      1500]
    # zero_list = [     200,    300,      4000,         500,      2000]
    all_time = 0
    speed_state = 0
    fatigue_time = 0
    i=0
    sta = 0
    pilao = 0
    while i < len(speed_list):
        all_time = all_time + speed_list[i]
        if all_time > 14400:
            if i == 0:
                number = number + 1
                pilao = pilao + all_time
                all_time = 0
            else:
                for indd,k in enumerate(zero_list[speed_state:i]):
                    if k > 1200:
                        all_time = 0
                        i = indd + speed_state
                        speed_state = speed_state + indd + 1
                        sta = 0
                        break
                if all_time != 0 :
                    if sta == 0 :
                        number = number + 1
                        pilao = pilao + all_time
                        sta = 1
                    else:
                        pilao = pilao + speed_list[i]
                        i = i + 1
                        continue
        i = i+1
    total_time = np.sum(speed_list)
    if total_time > 28800:
        number = number + 1
    if number > 2:
        number = 3

    return number, pilao, total_time


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


# 标准差分数
def std_score(row):
    std = row['标准差']
    ret = 100 - 3 * std
    if ret < 0:
        return 0
    else:
        return ret


# 超长怠速、怠速预热分数
def long_idle_score(row):
    time_temp = row['超长怠速累积时长']
    count_temp = row['超长怠速次数']
    if 0 <= time_temp < 100:
        time_score = 100
    elif 100 <= time_temp < 10000:
        time_score = 90
    else:
        time_score = 80 - time_temp / 100
        if time_score < 0:
            time_score = 0
    count_score = 100 - count_temp
    return (0.5 * count_score + 0.5 * time_score)


# 急加速，急减速分数
def speed_score(row):
    try:
        time_temp = row['急加速累积时长']
        count_temp = row['急加速次数']
    except:
        # 不存在“急加速XX”这一列，则是急减速
        time_temp = row['急减速累积时长']
        count_temp = row['急减速次数']
    time_score = 100 - 10 * time_temp
    count_score = 100 - 10 * count_temp
    if time_score < 0:
        time_score = 0
    if count_score < 0:
        count_score = 0
    return (0.5 * time_score + 0.5 * count_score)


# 超速行驶分数
def overspeed_score(row):
    # 时间换成分钟
    time_temp = row['超速累积时长'] / 60
    count_temp = row['超速次数']
    time_score = 100 - time_temp
    count_score = 100 - 3 * count_temp
    if time_score < 0:
        time_score = 0
    if count_score < 0:
        count_score = 0
    return (0.5 * time_score + 0.5 * count_score)


# 疲劳驾驶分数
def fatigue_score(row):
    time_temp = row['疲劳驾驶累计时长']
    count_temp = row['疲劳驾驶次数']
    if time_temp == 0:
        time_score = 100
    else:
        time_score = 0
    count_score = 100 - 15 * count_temp
    if count_score < 0:
        count_score = 0
    return (0.5 * time_score + 0.5 * count_score)


# 熄火滑行分数
def coasting_score(row):
    # 先换成分钟
    time_temp = row['熄火滑行累积时长'] / 60
    count_temp = row['熄火滑行次数']
    if time_temp <= 1:
        time_score = 100
    else:
        time_score = 100 - 10 * time_temp
        if time_score < 0:
            time_score = 0
    count_score = 100 - 20 * count_temp
    if count_score < 0:
        count_score = 0
    return (0.5 * count_score + 0.5 * time_score)



if __name__ == "__main__":
    """
    路径最后需要以"\\"结尾
    """
    out_dataset_1_path = "F:\Szu\out_data_1\\"
    out_dataset_1_dirs = os.listdir(out_dataset_1_path)

    for out_dataset_1_dir in out_dataset_1_dirs:

        out = pd.read_csv(r"F:\Szu\table1.csv")

        date_path = out_dataset_1_path + out_dataset_1_dir + "\\"
        date_dirs = os.listdir(date_path)
        # print(date_dirs)
        for date_dir in date_dirs:
            # 判断是否是文件夹
            if not(os.path.isfile(date_path + date_dir)):
                date_file_list = os.listdir(date_path + date_dir)  # 各日期子文件夹下的csv文件
                over_speed_count = []  # over_speed
                over_speed_time = []  # time_op
                urgent_acc_count = []  # urg_acc
                urgent_acc_time = []  # urg_acc_time
                rapidly_dec_count = []  # rap_dec
                rapidly_dec_time = []  # rap_dec_time
                coasting_weo_count = [] # coasting_with_engine_off熄火滑行次数
                coasting_weo_time = []  # cweo_time
                std = []  # 速度标准差

                fatigue_driving_count = []
                fatigue_driving_time = []

                date = date_dir.split('_')[-1]  # 文件的日期
                # 日期有些是8-4，有些是08-04，需处理成统一的08-04这种形式
                date1 = date.split('-')[0]
                date2 = date.split('-')[1]
                if len(date1) == 1:
                    date1 = '0' + date.split('-')[0]
                if len(date2) == 1:
                    date2 = '0' + date.split('-')[1]
                date = date1 + '-' + date2

                for date_file in date_file_list:
                    print(date_file)

                    df = pd.read_csv(date_path + date_dir + "\\" + date_file)
                    df = judge_over_speed(df)
                    df = add_rapidly_deceleration_label(df)
                    df = add_urgent_acceleration_label(df)
                    df = add_coasting_with_engine_off(df)
                    over_speed_count.append(df['over_speed'].sum())
                    over_speed_time.append(df['time_op'].sum())

                    urgent_acc_count.append(df['urg_acc'].sum())
                    urgent_acc_time.append(df['urg_acc_time'].sum())

                    rapidly_dec_count.append(df['rap_dec'].sum())
                    rapidly_dec_time.append(df['rap_dec_time'].sum())

                    coasting_weo_count.append(df['coasting_with_engine_off'].sum())
                    coasting_weo_time.append(df['cweo_time'].sum())

                    # a是疲劳驾驶次数，b是疲劳驾驶时长，c是总驾驶时间
                    a, b, c = fatigue_driving(date_path + date_dir + "\\" + date_file.split('.')[0])
                    fatigue_driving_count.append(a)
                    fatigue_driving_time.append(b)

                    temp_std = df['gps_speed'].std()
                    if pd.isna(temp_std): # 如果标准差是0，会填入nan
                        std.append(0)
                    else:
                        std.append(df['gps_speed'].std())

                out.loc[date] = [np.array(std).mean(), np.array(urgent_acc_time).sum(), np.array(urgent_acc_count).sum(), np.array(rapidly_dec_time).sum(), np.array(rapidly_dec_count).sum(), np.array(over_speed_time).sum(), np.array(over_speed_count).sum(), np.array(fatigue_driving_time).sum(), np.array(fatigue_driving_count).sum(), np.array(coasting_weo_time).sum(), np.array(coasting_weo_count).sum()]
            out_name = date_dir.split('_')[0] # 输出名称是车名

            if_dir_exists(r"F:\Szu\final1\\")

            if len(out) != 0:
                out['date'] = out.index
                out['sort'] = out.apply(lambda x: sort_date(x['date']), axis=1)
                out = out.sort_values(by='sort')
                # 只输出需要的值，用来排序的中间值不输出
                out = out[['标准差', '急加速累积时长', '急加速次数', '急减速累积时长', '急减速次数', '超速累积时长', '超速次数', '疲劳驾驶累计时长', '疲劳驾驶次数', '熄火滑行次数', '熄火滑行累积时长']]

                # 计算分值
                # 标准差分值，默认100
                try:
                    final_out = out.copy()
                    final_out['std_score'] = 100
                    final_out['std_score'] = final_out.apply(lambda row: std_score(row), axis=1)
                    # 急加速分数
                    final_out['urgent_acc_score'] = 100
                    final_out['urgent_acc_score'] = final_out.apply(lambda row: speed_score(row), axis=1)
                    # 急减速分数
                    final_out['rapidly_dec_score'] = 100
                    final_out['rapidly_dec_score'] = final_out.apply(lambda row: speed_score(row), axis=1)
                    # 超速分数
                    final_out['overspeed_score'] = 100
                    final_out['overspeed_score'] = final_out.apply(lambda row: overspeed_score(row), axis=1)
                    # 疲劳驾驶分数
                    final_out['fatigue_score'] = 100
                    final_out['fatigue_score'] = final_out.apply(lambda row: fatigue_score(row), axis=1)
                    # 熄火滑行分数
                    final_out['coasting_score'] = 100
                    final_out['coasting_score'] = final_out.apply(lambda row: coasting_score(row), axis=1)

                    final_out.to_csv(r"F:\Szu\final1\\" + out_name + '.csv')
                    # final_out.to_json(r"F:\Szu\final1\\" + out_name + '.json')
                except Exception as e:
                    print(e)
                    print(date_dir)

            else:
                # 计算分值
                try:
                    """表格无内容，保持列一致"""
                    final_out = out.copy()
                    final_out['std_score'] = 100
                    final_out['urgent_acc_score'] = 100
                    final_out['rapidly_dec_score'] = 100
                    final_out['overspeed_score'] = 100
                    final_out['fatigue_score'] = 100
                    final_out['coasting_score'] = 100

                    final_out.to_csv(r"F:\Szu\final1\\" + out_name + '.csv')
                    # final_out.to_json(r"F:\Szu\final1\\" + out_name + '.json')
                except Exception as e:
                    print(e)
                    print(date_dir)


    out_dataset_0_path = "F:\Szu\out_data_0\\"
    out_dataset_0_dirs = os.listdir(out_dataset_0_path)
    for out_dataset_0_dir in out_dataset_0_dirs:


        out = pd.read_csv(r"F:\Szu\final1\\" + out_dataset_0_dir + ".csv")
        out.index = out['Unnamed: 0']
        out = out[['标准差', '急加速累积时长', '急加速次数', '急减速累积时长', '急减速次数', '超速累积时长', '超速次数']].copy()
        out['超长怠速累积时长'] = 0
        out['超长怠速次数'] = 0
        out['怠速预热累积时长'] = 0
        out['怠速预热次数'] = 0

        date_path = out_dataset_0_path + out_dataset_0_dir + "\\"
        date_dirs = os.listdir(date_path)
        for date_dir in date_dirs:
            # 判断是否是文件夹
            if not (os.path.isfile(date_path + date_dir)):
                date_file_list = os.listdir(date_path + date_dir)  # 各日期子文件夹下的csv文件

                long_idle_count = []  # long_idle
                long_idle_time = []  # long_idle_time
                pre_heating_count = []  # pre_heating
                pre_heating_time = []  # pre_heating_time

                date = date_dir.split('_')[-1]  # 文件的日期
                # 日期有些是8-4，有些是08-04，需处理成统一的08-04这种形式
                date1 = date.split('-')[0]
                date2 = date.split('-')[1]
                if len(date1) == 1:
                    date1 = '0' + date.split('-')[0]
                if len(date2) == 1:
                    date2 = '0' + date.split('-')[1]
                date = date1 + '-' + date2

                for date_file in date_file_list:
                    print(date_file)
                    df = pd.read_csv(date_path + date_dir + "\\" + date_file)
                    df = add_long_idle_label(df)
                    long_idle_count.append(df['long_idle'].sum())
                    long_idle_time.append(df['long_idle_time'].sum())

                    pre_heating_count.append(df['pre_heating'].sum())
                    pre_heating_time.append(df['pre_heating_time'].sum())



                out.loc[date, ['超长怠速累积时长', '超长怠速次数', '怠速预热累积时长', '怠速预热次数']] = [np.array(long_idle_time).sum(), np.array(long_idle_count).sum(), np.array(pre_heating_time).sum(), np.array(pre_heating_count).sum()]

            out_name = date_dir.split('_')[0]
            if_dir_exists(r"F:\Szu\final2\\")
            out = out.fillna(0) #有些日期有怠速情况，但无超速、急加等情况


            if len(out) != 0:
                out['date'] = out.index
                out['sort'] = out.apply(lambda x: sort_date(x['date']), axis=1)
                out = out.sort_values(by='sort')
                # 只输出需要的值，用来排序的中间值不输出
                out = out[['标准差', '急加速累积时长', '急加速次数', '急减速累积时长', '急减速次数', '超速累积时长', '超速次数', '超长怠速累积时长', '超长怠速次数', '怠速预热累积时长', '怠速预热次数']]

                # 计算分值
                # 标准差分值，默认100
                try:
                    final_out = out.copy()
                    final_out['std_score'] = 100
                    final_out['std_score'] = final_out.apply(lambda row: std_score(row), axis=1)
                    # 急加速分数
                    final_out['urgent_acc_score'] = 100
                    final_out['urgent_acc_score'] = final_out.apply(lambda row: speed_score(row), axis=1)
                    # 急减速分数
                    final_out['rapidly_dec_score'] = 100
                    final_out['rapidly_dec_score'] = final_out.apply(lambda row: speed_score(row), axis=1)
                    # 超速分数
                    final_out['overspeed_score'] = 100
                    final_out['overspeed_score'] = final_out.apply(lambda row: overspeed_score(row), axis=1)
                    # 超长怠速分数
                    final_out['long_idle_score'] = 100
                    final_out['long_idle_score'] = final_out.apply(lambda row: long_idle_score(row), axis=1)
                    # 怠速预热分数
                    final_out['pre_heating_score'] = 100
                    final_out['pre_heating_score'] = final_out.apply(lambda row: long_idle_score(row), axis=1)

                    final_out.to_csv(r"F:\Szu\final2\\" + out_name + '.csv')
                    # final_out.to_json(r"F:\Szu\final2\\" + out_name + '.json')
                except Exception as e:
                    print(e)
                    print(date_dir)

            else:
                final_out = out.copy()
                final_out['std_score'] = 100
                final_out['urgent_acc_score'] = 100
                final_out['rapidly_dec_score'] = 100
                final_out['overspeed_score'] = 100
                final_out['long_idle_score'] = 100
                final_out['pre_heating_score'] = 100

                final_out.to_csv(r"F:\Szu\final2\\" + out_name + '.csv')
                # final_out.to_json(r"F:\Szu\final2\\" + out_name + '.json')