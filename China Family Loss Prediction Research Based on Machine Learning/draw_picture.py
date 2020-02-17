#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.2
@Created on 2019-3-5 10:40
@Author:ChileWang
@algorithm：
画图
"""
from data_train_and_predict import *
import matplotlib.pyplot as plt
import random
import json
from sklearn.metrics import confusion_matrix

# plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
image_to_save = 'image/'  # 图片存放文件夹
nominal_dividing_bucket_meaning = 'JSON/nominal_dividing_bucket/'  # 标称型分桶字典的具体含义
numeric_dividing_bucket_meaning = 'JSON/numeric_dividing_bucket/'  # 数值型分桶字典的具体含义


def produce_color(num):
    """
    随机生成num个不同的颜色
    :param num: 数量
    :return:
    """
    color_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color_list = []
    while len(color_list) < num:
        color = ""
        for j in range(6):
            color += color_arr[random.randint(0, 14)]
        color = "#" + color
        if ~(color in color_list):  # 若该颜色不在该列表，则添加
            color_list.append(color)
    return color_list


def draw_bar_chart(name_list, heights, x_label, y_label, title):
    """
    绘制条形图
    :param name_list: 名字列表
    :param heights:　名字对应的数值列表
    :param x_label:　X轴名字
    :param y_label:　Y轴名字
    :param title:　表名
    :return:
    """
    plt.ion()  # 显示图片
    # 设置条形码的相应位置
    positions = [i for i in range(len(name_list))]
    plt.bar(positions, heights, color=produce_color(1), alpha=0.8, align='center', edgecolor='white')
    # 设置坐标轴名称
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(positions, name_list)  # 设置数据分类名称
    plt.title(title)  # 设置标题名称
    # 设置数字标签
    for x, y in zip(positions, heights):
        """
        其中，a, b+0.05表示在每一柱子对应x值、y值上方0.05处标注文字说明， 
        '%.0f' % b,代表标注的文字，即每个柱子对应的y值，
         ha='center', va= 'bottom'代表horizontalalignment（水平对齐）、
         verticalalignment（垂直对齐）的方式，fontsize则是文字大小。
        """
        plt.text(x, y+0.05, '%.0f' % y, ha='center', va='bottom')
    plt.legend([y_label], loc='best')  # 图例
    # plt.grid()  # 网格线
    plt.savefig(image_to_save + title)  # 保存图片
    plt.pause(1)  # 显示秒数
    plt.close()


def draw_pie_chart(name_list, name_rate, title):
    """
    生成饼图
    :param name_list: 名字列表
    :param name_rate: 名字对应的百分比
    :param title: 表名
    :return:
    """

    plt.figure(figsize=(12, 6))
    plt.ion()  # 显示图片

    color_list = produce_color(len(name_list))  # 生成互不相同的颜色
    explode_list = [0 for i in range(len(name_list))]  # 构造饼图分割表
    if len(explode_list) > 0:
        explode_list[-1] = 0.15  # 最后一个名字分割出来
    plt.pie(name_rate, labels=name_list, colors=color_list, explode=explode_list, autopct='%3.1f%%',
            shadow=False, startangle=90, pctdistance=0.6)
    plt.title(title)
    plt.legend(name_list, loc='best', bbox_to_anchor=(1.5, 1))
    # bbox_to_anchor控制左右上下移动， 第一个是控制左右，第二个控制上下， 越大越往右/上移动
    plt.savefig(image_to_save + title)  # 保存图片
    plt.pause(1)  # 显示秒数
    plt.close()


def draw_roc_line(positive_rate, negative_rate, y_label_list, auc_scorce):
    """
    绘制roc折线图
    根据样本标签统计出
    正负样本的数量，假设正样本数量为P，负样本数量为N；接下来，把横轴的刻度
    间隔设置为1/N，纵轴的刻度间隔设置为1/P；再根据模型输出的预测概率对样本进
    行排序（从高到低）；依次遍历样本，同时从零点开始绘制ROC曲线，每遇到一
    个正样本就沿纵轴方向绘制一个刻度间隔的曲线，每遇到一个负样本就沿横轴方
    向绘制一个刻度间隔的曲线，直到遍历完所有样本，曲线最终停在（1,1）这个
    点，整个ROC曲线绘制完成。
    :param positive_rate: １/正样本个数
    :param negative_rate: １/负样本个数
    :param y_label_list: 基于预测概率从高到底排序的真实标签列表
    :param auc_scorce: auc面积
    :return:
    """
    plt.ion()  # 显示图片
    # 从(0, 0)开始
    x = 0
    y = 0
    x_list = [0]  # 横坐标
    y_list = [0]  # 纵坐标
    for label in y_label_list:
        if label == 1:  # 正样本
            y += positive_rate
            x_list.append(x)
            y_list.append(y)
        else:  # 负样本
            x += negative_rate
            x_list.append(x)
            y_list.append(y)
    # 到(1, 1)结束
    x_list.append(1)
    y_list.append(1)
    plt.plot(x_list, y_list, "b--", linewidth=1, label='auc score = ' + str(auc_scorce))  # (X轴，Y轴，蓝色虚线，线宽度, 图例)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='best')  # 让图例显效
    plt.savefig(image_to_save + 'roc_curve')  # 保存图片
    plt.pause(5)  # 显示秒数
    plt.close()


def nominal_type_statistic_draw(data_set, col, title):
    """
    标称型数据类型占比统计并绘制相应饼图
    将标称型数据下的数据类型的占比进行统计，从大到小相加，当占比超过85%的时候，剩余的合并成一类，
    并返回合并形成一个集合，并绘制饼图
    :param data_set:
    :param col: 统计的列名
    :param title: 图表名称
    :return:
    """
    temp = data_set[[col]].copy()
    temp['col_count'] = 1
    temp = temp.groupby([col], as_index=False).count()  # 基于col排序,将col_count统计
    data_num = data_set.shape[0]  # 总行数
    temp['col_rate'] = temp.col_count.apply(lambda x: x / data_num)
    temp.sort_values(by="col_rate", ascending=False, inplace=True)  # 对占比从大到小排序
    temp.reset_index(inplace=True)
    #  将对应的列和起百分比压缩成一个按值从大到小的字典
    col_list = list(temp[col])
    col_rate_list = list(temp['col_rate'])
    col_name_list = []  # 存放这个特征下所有数字代表的真正含义
    meaning = col  # 特征的实际意义
    print('---------')
    print(col)
    if col != 'countyid':  # 不画countyid的饼图
        with open(nominal_dividing_bucket_meaning + col + '.json', 'r', encoding='UTF-8') as fr:  # 读取对应特征的json文件
            col_name_dict = dict(json.load(fr))  # 存放该特征的数字所对应的具体含义的字典
            meaning = col_name_dict['meaning']
        for col_key in col_list:
            col_name_list.append(col_name_dict[str(int(col_key))])

        col_dict = dict(zip(col_name_list, col_rate_list))
    else:
        col_dict = dict(zip(col_list, col_rate_list))

    fina_data_list = list()  # 饼图最终的名称对应列表
    threshold_rate = 0.85  # 阈值占比
    sum_rate = 0.0  # 累计占比
    name_rate = []  # 饼图的占比
    for key in col_dict.keys():
        if sum_rate >= threshold_rate:
            if col != 'countyid':
                fina_data_list.append('其他')  # 剩余的类别归并成一类，并命名为其他
            else:
                fina_data_list.append(key)  # 若是县区特征，则直接用数字表示
            name_rate.append(1 - sum_rate)
            break
        sum_rate += col_dict[key]
        name_rate.append(col_dict[key])
        fina_data_list.append(key)

    # 生成饼图
    # draw_pie_chart(fina_data_list, name_rate, meaning + title)
    return col_list


def draw_confuse_matrix(y_true, y_preb,  title='Confusion Matrix'):
    """

    :param y_true: 真实标签
    :param y_preb: 预测标签
    :param title: 表名
    :return:
    """
    cm = confusion_matrix(y_true, y_preb)
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    labels = list(set(y_true))
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def draw_nominal_and_roc_pic(model_name):
    """
    画标称型特征图和ROC曲线
    :return:
    """
    train_set = pd.read_csv(data_to_train_and_test + 'train_set.csv')
    test_set = pd.read_csv(data_to_train_and_test + 'test_set.csv')
    df_list = [train_set, test_set]  # 将训练集和测试集放入列表中
    df_list = year_partition(df_list, year_type)  # 将年份的分桶

    # 生成训练集和测试集的数量条形图
    name_list = ['train set', 'test set']
    heights = [train_set.shape[0], test_set.shape[0]]
    x_label = 'data set'
    y_label = 'amount of data set'
    title = 'amount of train set and test set'
    draw_bar_chart(name_list, heights, x_label, y_label, title)

    # 生成数值特征和标称特征的饼图和条形图
    name_list = ['nominal_type', 'numerical_type']
    name_rate = [len(nominal_type), len(numerical_type)]
    title = 'numerical_type and nominal_type'
    draw_pie_chart(name_list, name_rate, title)
    x_label = 'type of features'
    y_label = 'amount of features'
    draw_bar_chart(name_list, name_rate, x_label, y_label, title)

    # 标称型缺失值统一为-30
    for df in df_list:
        df[nominal_type] = df[nominal_type].fillna(-30)
    family_loss_set = df_list[0][df_list[0]['label'] == 1].copy()  # 抽出label为１的个人流失数据
    family_not_loss_set = df_list[0][df_list[0]['label'] == 0].copy()  # 抽出label为１的个人流失数据

    # 生成训练集label=1和label=0的饼图和条形图
    name_list = ['0', '1']
    name_rate = [df_list[0][df_list[0]['label'] == 0].shape[0], df_list[0][df_list[0]['label'] == 1].shape[0]]
    title = 'percentage of label type in train set'
    draw_pie_chart(name_list, name_rate, title)
    title = 'amount of label type in train set'
    x_label = 'type of labels'
    y_label = 'amount of labels'
    draw_bar_chart(name_list, name_rate, x_label, y_label, title)

    for nt in nominal_type:  # 对特定的标称型特征进行占比统计
        fina_key_list = nominal_type_statistic_draw(family_loss_set, nt, '在pid流失人群的占比情况')
        # 得到最终的占比统计类别与剩余类别并画出饼图　
        family_not_loss_list = nominal_type_statistic_draw(family_not_loss_set, nt, '在pid非流失人群的占比情况')
        # 得到最终的占比统计类别与剩余类别并画出饼图
        # 对训练集和测试集合相应的特征进行新的分类, 若类在于fina_key_list当中，则继续应用，否则将以得到的剩余类别将其分类
        df_list[0][nt] = df_list[0][nt].apply(lambda x: x if x in fina_key_list else fina_key_list[-1])
        df_list[1][nt] = df_list[1][nt].apply(lambda x: x if x in fina_key_list else fina_key_list[-1])

    # 数值型特征缺失值补充
    df_list = medium_fill(df_list, numerical_type)

    # 特征选择
    df_list = var_delete(df_list, nominal_type, numerical_type)  # 方差选择法
    df_list = select_best_chi2(df_list)  # 卡方验证选择法

    # 标称型独热编码
    columns = df_list[1].columns.values.tolist()  # 测试集不包含label
    new_nominal_type = list(set(columns) & set(nominal_type))  # 新的标称型列
    new_numerical_type = list(set(columns) & set(numerical_type))  # 新的数值型列
    df_list, new_nominal_type = one_hot_encode(df_list, new_nominal_type)  # 独热编码
    new_feature = new_nominal_type + new_numerical_type  # 新的特征列

    # PCA降维
    n_components = 5  # pca的维度
    new_pridictors = ['a' + str(i) for i in range(n_components)]  # 降维后的新特征
    pca_train_set = pd.DataFrame(pca_decompose(df_list[0][new_feature].values, n_components), columns=new_pridictors)  # DF转矩阵
    pca_test_set = pd.DataFrame(pca_decompose(df_list[1][new_feature].values, n_components), columns=new_pridictors)  # DF转矩阵
    # 将相应的id补上
    pca_train_set['label'] = df_list[0]['label']
    pca_train_set['pid'] = train_set['pid']
    pca_train_set['fid'] = train_set['fid']

    pca_test_set['pid'] = test_set['pid']
    pca_test_set['fid'] = test_set['fid']

    # 获取模型
    model = get_model(model_name)

    # 验证
    new_valid_set = train_test_split(pca_train_set, test_size=0.2, stratify=pca_train_set['label'], random_state=100)[1]
    x_valid = new_valid_set[new_pridictors]
    y_valid = new_valid_set['label']
    y_pred = model.predict(x_valid)
    y_pred_prob = model.predict_proba(x_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, y_pred_prob)
    print('Accuracy: %.4g' % accuracy_score(y_valid.values, y_pred))
    print('AUC Score (Train): %.4g' % auc_score)

    # 绘制roc曲线
    positive_rate = 1/(new_valid_set[new_valid_set['label'] == 1].shape[0])  # 正样本刻度
    negative_rate = 1/(new_valid_set[new_valid_set['label'] == 0].shape[0])  # 负样本刻度
    temp_dict = {'label': y_valid, 'pred_prob': y_pred_prob}
    temp = pd.DataFrame(temp_dict)
    temp.sort_values('pred_prob', inplace=True, ascending=False)  # 基于预测概率倒序排序
    draw_roc_line(positive_rate, negative_rate, list(temp['label']), auc_score)

    # 绘制混淆矩阵
    draw_confuse_matrix(list(y_valid), list(y_pred))
    pic = pd.crosstab(list(y_valid), list(y_pred), rownames=['True label'], colnames=['Predicted label'])
    print(pic)


def years_old_judge(row):
    """
    年龄分桶
    :param row:
    :return:
    """
    if row < 23:
        return '0~23岁'
    elif (row >= 23) and (row < 40):
        return '23~40岁'
    elif (row >= 40) and (row < 58):
        return '40~58岁'
    elif (row >= 58) and (row < 75):
        return '58~75岁'
    elif row >= 75:
        return '75岁以上'
    else:
        return '其他'


def total_funds_judge(row):
    """
    资金总额分桶
    :param row:
    :return:
    """
    if row < -50000:
        return '大额负债(负债大于5万)'
    elif (row >= -50000) and (row < 0):
        return '小额负债(负债0~5万)'
    elif (row >= 0) and (row < 150000):
        return '小额资产(0~15万)'
    elif (row >= 150000) and (row < 300000):
        return '中额资产(15~30万)'
    elif (row >= 300000) and (row < 1000000):
        return '大额资产(30~100万)'
    elif row >= 1000000:
        return '超大额资产(100万)'
    else:
        return '其他'


def family_size_judge(row):
    """
    家庭规模分桶
    :param row:
    :return:
    """
    if row < 3:
        return '1~2人'
    elif (row >= 3) and (row < 5):
        return '3~4人'
    elif (row >= 5) and (row < 8):
        return '5~7人'
    elif row >= 8:
        return '大于8人'
    else:
        return '其他'


def numeric_type_partition(df_list, numeric_dict):
    """
    对数值类型分桶
    :param df_list: 训练集与验证集与测试集
    :param numeric_dict: 数值型字典，key为名字，values为分桶区间列表
    :return:
    """
    for i in range(len(df_list)):
        for col in year_type:
            df_list[i][col] = df_list[i][col].apply(year_judge)
    return df_list


def direct_dividing_rule(row, range_list=[0, 1000, 2000]):
    """
    直接分桶规则，按照范围划分
    :param range_list: 范围列表
    :return:
    """
    if (row >= range_list[0]) and (row < range_list[1]):
        return str(range_list[0]) + '~' + str(range_list[1])
    elif (row >= range_list[1]) and (row < range_list[2]):
        return str(range_list[1]) + '~' + str(range_list[2])
    else:
        return '大于' + str(range_list[2])


def direct_dividing_bucket(df_list, direct_numeric_type_dict):
    """
    :param df_list: 训练集和测试集
    :param direct_numeric_type_dict: 直接分桶的数值型特征字典 key : value -> feature_name: range_list
    :return:
    """
    for df in df_list:
        for key in direct_numeric_type_dict.keys():
            range_list = direct_numeric_type_dict[key]  # 范围列表
            df[key] = df[key].apply(direct_dividing_rule, range_list)  # 对家庭规模分桶
    return df_list


def dividing_bucket_with_medium_rule(row, med_num_list=[1000, 500]):
    """
    中位数加减某个数，确定范围
    :param row:
    :param med_num_list: 中位数 和 商议好的相加减的数
    :return:
    """
    med = med_num_list[0]
    num = med_num_list[1]
    if row < med - num:
        return '0~' + str(med - num)
    elif (row >= med - num) and (row < med + num):
        return str(med-num) + '~' + str(med + num)
    elif row >= med + num:
        return '大于' + str(med + num)
    else:
        return '其他'


def dividing_bucket_with_medium(df_list, numeric_type_dict_with_med):
    """
    :param df_list: 训练集和测试集
    :param numeric_type_dict_with_med: 利用中位数分桶的数值型特征字典 key : value -> feature_name: num
    :return:
    """
    for df in df_list:
        for key in numeric_type_dict_with_med.keys():
            print(df[key].describe())
            print('---------')
            med = df[key].median()  # 该列中位数
            num = numeric_type_dict_with_med[key]  # 中位数加减的那个数
            med_num_list = [med, num]

            df[key] = df[key].apply(dividing_bucket_with_medium_rule, med_num_list)
    return df_list


def numeric_type_statistic_draw(data_set, col, title):
    """
    数值型数据类型占比统计并绘制相应饼图
    将标称型数据下的数据类型的占比进行统计，并绘制饼图
    :param data_set:
    :param col: 统计的列名
    :param title: 图表名称
    :return:
    """
    temp = data_set[[col]].copy()
    temp['col_count'] = 1
    temp = temp.groupby([col], as_index=False).count()  # 基于col排序,将col_count统计
    data_num = data_set.shape[0]  # 总行数
    temp['col_rate'] = temp.col_count.apply(lambda x: x / data_num)
    temp.sort_values(by="col_rate", ascending=False, inplace=True)  # 对占比从大到小排序
    temp.reset_index(inplace=True)
    #  将对应的列和起百分比压缩成一个按值从大到小的字典
    col_list = list(temp[col])
    col_rate_list = list(temp['col_rate'])
    if col == 'total_funds':
        title = '个人资产' + title
    elif col == 'familysize':
        title = '家庭规模' + title
    else:
        title = '年龄段' + title
    # 生成饼图
    draw_pie_chart(col_list, col_rate_list, title)


def draw_numeric_pic():
    """
    用来画数值型特征图
    :return:
    """
    numerical_money_type = ['qk601', 'fe601', 'fe802', 'fe903', 'ff2', 'indinc', 'land_asset', 'total_asset',
                            'expense', 'fproperty']  # 数值型金钱特征
    numerical_income_type = ['qk601', 'fe601', 'fe802', 'fe903', 'ff2', 'indinc', 'land_asset', 'total_asset',
                            'fproperty']  # 数值型收入特征
    train_set = pd.read_csv(data_to_train_and_test + 'train_set.csv')
    test_set = pd.read_csv(data_to_train_and_test + 'test_set.csv')
    df_list = [train_set, test_set]  # 将训练集和测试集放入列表中
    for df in df_list:
        for col in numerical_money_type:
            df[col] = df[col].apply(lambda x: 0 if x == -8 else x)  # 将0替代-8

    # 数值型特征缺失值用每一列的中位数去填充
    df_list = medium_fill(df_list, numerical_type)

    # for col in numerical_money_type:
    #     print(df_list[0][col].unique())
    for df in df_list:
        df['total_funds'] = 0
        for col in numerical_income_type:
            df['total_funds'] += df[col]
        df['total_funds'] -= df['expense']  # 减掉自己支出等于资金总和
        # print('------------')
        # print(df['total_funds'].describe())
    for df in df_list:
        df['qa1age'] = df['qa1age'].apply(years_old_judge)  # 对年龄分桶
        df['familysize'] = df['familysize'].apply(family_size_judge)  # 对家庭规模分桶
        df['total_funds'] = df['total_funds'].apply(total_funds_judge)  # 对家庭资产分桶
    print(df_list[0]['total_funds'].head(5))
    print(df_list[0]['familysize'].head(5))
    print(df_list[0]['qa1age'].head(5))
    # # 按照范围直接分桶
    # with open(numeric_dividing_bucket_meaning + 'numeric_type.json', 'r', encoding='UTF-8') as fr:
    #     # 读取直接分桶的数值型特征 key : value -> feature_name: range_list
    #     direct_numeric_type_dict = dict(json.load(fr))
    # df_list = direct_dividing_bucket(df_list, direct_numeric_type_dict)
    #
    # # 利用中位数加减商议好的某个数进行分桶
    # with open(numeric_dividing_bucket_meaning + 'numeric_type_with_med.json', 'r', encoding='UTF-8') as fr:
    #     # 读取利用中位数分桶的数值型特征 key : value -> feature_name: medium
    #     numeric_type_with_med_dict = dict(json.load(fr))
    # df_list = dividing_bucket_with_medium(df_list, numeric_type_with_med_dict)
    #
    family_loss_set = df_list[0][df_list[0]['label'] == 1].copy()  # 抽出label为１的个人流失数据
    family_not_loss_set = df_list[0][df_list[0]['label'] == 0].copy()  # 抽出label为１的个人流失数据
    draw_pie_col = ['total_funds', 'familysize', 'qa1age']  # 对这三个特征画饼图
    for col in draw_pie_col:
        numeric_type_statistic_draw(family_loss_set, col, '在pid流失人群的占比情况')
        # 得到最终的占比统计类别与剩余类别并画出饼图　
        numeric_type_statistic_draw(family_not_loss_set, col, '在pid非流失人群的占比情况')


if __name__ == '__main__':
    draw_nominal_and_roc_pic('GBDT')
    # draw_numeric_pic()
