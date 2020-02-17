# 运行程序说明
## 0.运行环境 
Anaconda python 3.7.1
## 1.第三方库要求
除Anaconda内含的库外，绘制行车路线图需要安装python的第三方库folium（版本0.8.3）
## 2.运行顺序 
3-4-5-9-6-7-9-8
## 3.DataProcessing.py 
数据预处理，为源文件添加时间戳，然后先将文件拆分出速度大于0的子路程，再按日期提取有速度的子路程，接着同样处理速度小于0的子路程。
## 4.Question_one.py 
统计各车不同日期的行车里程、平均速度、急加速急减速情况
## 5.Question_one_draw.py 
绘制每辆车在经纬度坐标系下的路线
## 6.Question_two.py 
挖掘车辆的不良驾驶行为并评分
## 7.Question_two_dig.py 
输出一个大表，存放各车存在的不良驾驶行为以及各车最终的安全评价得分，其中各行车安全评价指标的权重可通过AHP.py获得
## 8.Question_three.py 
输出一个表，给所有车辆的驾驶行为评总分，其中各车的安全评价指标、节能评价指标的权重可通过AHP.py获得
## 9.AHP.py
层次分析法计算权重
## 10.weather.py
对天气数据进行一些统计分析

# 文件夹
## 1.Question_one
Question_one.py、Question_one.py所在目录
## 2.Question_two
Question_two.py、Question_two_dig.py所在目录
## 3.Question_three
Question_three.py所在目录


# 文件
## 1.table1.csv 
建模所需的一个表格模板