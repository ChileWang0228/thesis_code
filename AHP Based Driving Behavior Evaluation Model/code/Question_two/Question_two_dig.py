# !/usr/bin/env python
# -*- conding: utf-8 -*-
'''
运行环境：Anaconda python 3.7.1
@Author：Cheng
@Created on：2019/4/15 19:00
@Algorithm:
'''
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


def get_behavior_label(file, target):
    """用Question_two.py输出的结果
    :param file: 一辆的车行驶行为的统计表，绝对路径
    :param target: 指标名称
    :return: 1 or 0
    """
    df = pd.read_csv(file)
    df = df.loc[df[target] > 0]
    if len(df) > 0:
        return 1
    else:
        return 0


def get_score(file):
    """
    打分
    :param file: 文件路径
    :return: 分数
    """
    weights = [[0.08724146],
                [0.17881193],
                [0.22931698],
                [0.08724146],
                [0.17881193],
                [0.23857624]]  # 打分权重
    df = pd.read_csv(file)
    val = df[['std_score', 'urgent_acc_score', 'rapidly_dec_score', 'overspeed_score', 'fatigue_score', 'coasting_score']].mean().values
    return np.dot(val, weights)[0]



if __name__ == "__main__":
    """不良驾驶行为
    urgent_acc_score:急加速
    rapidly_dec_score:急减速
    overspeed_score:超速
    fatigue_time_score:疲劳驾驶
    coasting_count_score:熄火滑行
    long_idle_score:超长怠速
    pre_heating_score:怠速预热
    """
    columns = ['vehicle', 'urgent_acc', 'rapidly_dec', 'overspeed', 'fatigue_drive', 'coasting',
               'long_idle', 'pre_heating', 'score_security']
    out_table = pd.DataFrame(np.zeros((450, len(columns))), columns=columns)
    file_path = r"F:\Szu\final1\\"
    file_list = os.listdir(file_path)
    count = 0

    for file in file_list:
        print(file)
        out_label = [file.split('.')[0]]
        need_columns = ['急加速次数', '急减速次数', '超速次数', '疲劳驾驶次数', '熄火滑行次数']

        for i in need_columns:
            try:
                out_label.append(get_behavior_label(file_path + file, i))
            except Exception:
                out_label.append(0)
        need_columns_2 = ['超长怠速次数', '怠速预热次数']
        file_path_2 = r"F:\Szu\final2\\" + file

        for j in need_columns_2:
            try:
                out_label.append(get_behavior_label(file_path + file, j))
            except Exception:
                out_label.append(0)

        # 标记不良驾驶行为
        out_table.loc[count, columns[: -1]] = out_label

        # 打分
        out_table.loc[count, 'score_security'] = get_score(file_path + file)

        count += 1
    out_path = r"F:\Szu\Question_two\\"
    if if_dir_exists(out_path):
        out_table.to_csv(out_path + "Question_two.csv")