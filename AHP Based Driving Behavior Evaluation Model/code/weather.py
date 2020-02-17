# !/usr/bin/env python
# -*- conding: utf-8 -*-
'''
运行环境：Anaconda python 3.7.1
@Author：Cheng
@Created on：2019/4/22 19:24
@Algorithm:
'''
import pandas as pd


def heavy_rain(x):
    if "转大雨" in x:
        return 1
    elif x == "大雨":
        return 1
    else:
        return 0


def cloudy(x):
    if "转多云" in x:
        return 1
    elif x == "多云":
        return 1
    else:
        return 0


def sunny(x):
    if "转晴" in x:
        return 1
    elif x == "晴":
        return 1
    else:
        return 0


def light_rain(x):
    if "转小雨" in x:
        return 1
    elif x == "小雨":
        return 1
    else:
        return 0


w = pd.read_csv("F:\Szu\weather.csv", encoding='gbk')
l = len(w)
w['heavy_rain'] = w.conditions.apply(lambda x: heavy_rain(x))
print('大雨\n', w['heavy_rain'].value_counts()[1] * 100 / l)

w['cloudy'] = w.conditions.apply(lambda x: cloudy(x))
print('多云\n', w['cloudy'].value_counts()[1] * 100 / l)

w['sunny'] = w.conditions.apply(lambda x: sunny(x))
print('晴\n', w['sunny'].value_counts()[1] * 100 / l)

w['light_rain'] = w.conditions.apply(lambda x: light_rain(x))
print('小雨\n', w['light_rain'].value_counts()[1] * 100 / l)
