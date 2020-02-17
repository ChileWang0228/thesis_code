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
import folium

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


if __name__ == "__main__":
    car_dir = r"F:\Szu\szu\\"
    upload_car_list = ['AA00002', 'AB00006', 'AD00003', 'AD00013', 'AD00053', 'AD00083', 'AD00419', 'AF00098', 'AF00131', 'AF00373']

    out_table = pd.DataFrame(np.zeros((10, 6)), columns=["urg_acc", "urg_acc_time", "rap_dec", "rap_dec_time", "driving_mileage", "avg_speed"])
    out_table.index = upload_car_list
    for car_name in upload_car_list:
        '''提交附表中 10 辆车每辆车每条线路在经纬度坐标系下的
        运输线路图及对应的行车里程、平均行车速度、急加速急减速情况。'''
        df = pd.read_csv(car_dir + car_name + ".csv")

        lat_avg = df['lat'].mean()
        lng_avg = df['lng'].mean()
        mydata1 = list(df.loc[:, ['lat', 'lng']].values)
        mydata2 = [(i, j) for i, j in mydata1]
        # oneUserMap = folium.Map(location=[40.0764, 116.2786], zoom_start=4)
        # 以行驶路线经纬度均值为图片展现中心，放大倍数默认6倍
        oneUserMap = folium.Map(location=[lat_avg, lng_avg], zoom_start=6)
        folium.PolyLine(mydata2).add_to(oneUserMap)

        out_dir = r"F:\Szu\Question_one\draw\\"
        if if_dir_exists(out_dir):
            print(car_name)
            oneUserMap.save(out_dir + car_name + ".html")  # 行车轨迹保存为html文件

