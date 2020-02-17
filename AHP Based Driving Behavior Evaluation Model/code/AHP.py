# -*- coding: utf-8 -*-
"""
编译环境：Anaconda python 3.6.8
@Created on 2019-4-7 22:53
@Author:ChileWang
@algorithm：
层次分析法:
详情请看同级目录PPT
"""
import numpy as np
import pandas as pd

RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def get_w(compare_matrix):
    print('---------------')
    n = compare_matrix.shape[0]  # 计算出阶数

    # 让对比矩阵的每一列归一化
    a_axis_0_sum = compare_matrix.sum(axis=0)  # 按列求和
    new_compare_matrix = compare_matrix / a_axis_0_sum

    #  求最大特征值的特征向量
    b_axis_1_sum = new_compare_matrix.sum(axis=1)  # 按行求和
    # print(b_axis_1_sum)
    w = b_axis_1_sum / b_axis_1_sum.sum(axis=0)  # 归一化处理(所求特征向量)

    # 计算AW
    AW = compare_matrix * w
    # 近似的最大特征值
    lambda_max = sum(AW / (n * w))

    # 判断对比矩阵一致性
    CI = (lambda_max - n) / (n - 1)
    CR = CI / RI_dict[n]
    cr_value = np.array(CR)[0][0]  # 提取出矩阵的CR值
    if CR < 0.1:
        print(round(cr_value, 3))
        print('满足一致性')
        print('近似最大特征值:%s' % lambda_max)
        print('特征向量:%s' % w)
        return w
    else:
        print(round(cr_value, 3))
        print('不满足一致性，请进行修改')


def main(compare_matrix):
    if type(compare_matrix) is np.matrix:
        return get_w(compare_matrix)
    else:
        print('请输入mat对象')


if __name__ == '__main__':
    compare_matrix_security = np.mat([
        [1, 1/2, 1/3, 1, 1/2, 1/3],
        [2, 1, 1, 2, 1, 1/2],
        [3, 1, 1, 3, 1, 1],
        [1, 1/2, 1/3, 1, 1/2, 1/3],
        [2, 1, 1, 2, 1, 1/2],
        [3, 2, 1, 3, 1/2, 1]
    ])
    
    # 权重
    ws = main(compare_matrix_security)
    
    compare_matrix_green = np.mat([
        [1, 1/3, 1/2, 1/3, 1/2, 1/3],
        [3, 1, 2, 1, 2, 1],
        [2, 1/2, 1, 1/2, 1, 1/2],
        [3, 1, 2, 1, 2, 1],
        [2, 1/2, 1, 1/2, 1, 1/2],
        [3, 1, 2, 1, 2, 1]
    ])
    # 权重
    wg = main(compare_matrix_green)

    score_table = pd.read_csv(r"F:\Szu\Question_three\score.csv")

    security_table = score_table[['vehicle', 'speed_std', 'urgent_acc', 'rapidly_dec', 'overspeed', 'fatigue_drive', 'coasting']]

    vehicle = "AA00001"
    car_security_score = np.array(security_table[security_table['vehicle']==vehicle])[:, 1:]
    security_score = np.dot(car_security_score, ws)

    green_table = score_table[['vehicle', 'speed_std', 'long_idle', 'urgent_acc', 'rapidly_dec', 'overspeed', 'pre_heating']]
    car_green_score = np.array(green_table[green_table['vehicle']=='AA00001'])[:, 1:]
    green_score = np.dot(car_green_score, wg)


    total_score = 0.5 * security_score + 0.5 * green_score
    print("Score of vehicle {} : {}".format(vehicle, total_score[0, 0]))
