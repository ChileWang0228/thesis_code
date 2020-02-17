#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.2
@Created on 2019-1-30 1:43
@Author:ChileWang
@algorithm：
模型训练与预测
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, \
    precision_score, confusion_matrix, precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import pickle
import xgboost as xgb

import matplotlib.pyplot as plt

image_to_save = 'image/'  # 图片存放文件夹
data_to_train_and_test = 'data_to_train_and_test/'  # 训练集和测试集csv文件存放地址
numerical_type = ['qa1age', 'qk601', 'fe601', 'fe802', 'fe903', 'ff2', 'indinc',
                  'land_asset', 'total_asset', 'expense', 'familysize', 'fproperty']  # 数值型特征
nominal_type = ['provcd', 'countyid', 'urban', 'gender', 'qd3', 'qe1_best', 'qe507y',
                'qe211y', 'qp3', 'wm103', 'wn2', 'birthy_best', 'alive_a_p', 'tb3_a_p',
                'tb4_a_p', 'alive_a_f', 'alive_a_m', 'tb6_a_f', 'tb6_a_m']  # 标称型特征
year_type = ['qe507y', 'qe211y', 'birthy_best']  # 年份型标称变量, 对其进行分桶
# pd.set_option('display.max_columns', None)  # 显示所有列


def one_hot_encode(encode_df_list, encode_list):
    """
    对标称型变量进行独热编码
    :param encode_df_list: 数据集
    :param encode_list: 需要独热编码的列
    :return:
    """
    feature = []
    for i in range(len(encode_df_list)):
        for col in encode_list:
            one_hot_columns = encode_df_list[i][col].unique()
            temp_df = pd.get_dummies(encode_df_list[i][col].replace('null', np.nan))  # 独热编码
            temp_df.columns = one_hot_columns
            encode_df_list[i][one_hot_columns] = temp_df  # 合并到原来的数据集
            if i == 1:
                feature.extend(one_hot_columns)  # 新生成的标称型特征
    return encode_df_list, feature


def medium_fill(df_list, feature):
    """
    对数值型特征用中位数进行缺失值填充
    :param df_list:  训练集与验证集与测试集
    :param feature: 数值型特征
    :return:
    """

    for i in range(len(df_list)):
        for col in feature:
            # 中位数填充
            med = df_list[i][col].median()
            df_list[i][col].fillna(med, inplace=True)
    return df_list


def var_delete(df_list, non_feature, num_feature):
    """
    方差删除法
    对标称型特征进行方差选择法，去掉取值变化小的特征
    :param df_list:
    :param non_feature: 适用于标称型特征
    :param num_feature: 适用于标称型特征
    :return:
    """
    var_del_model = VarianceThreshold(threshold=(.8 * (1 - .8)))
    var_del_model.fit_transform(df_list[0][non_feature])
    selected_index = var_del_model.get_support(indices=True)
    new_columns = []
    for index in selected_index:
        new_columns.append(non_feature[index])
    new_columns.extend(num_feature)  # 构造新的特征
    for i in range(len(df_list)):
        tem_df = df_list[i][new_columns].copy()  # 重新构造新的数据集
        if i < 1:
            tem_df['label'] = df_list[i]['label']
        df_list[i] = tem_df
    return df_list


def select_best_chi2(df_list):
    """
    利用卡方验证选择最佳的前ｋ个特征
    利用sklearn的库会让DataFrame变成二维列表
    :param df_list:
    :return:
    """
    columns = df_list[-1].columns.values.tolist()  # 测试集不包含label
    select_model = SelectKBest(chi2, k=25)  # 卡方验证选择模型
    X = MinMaxScaler().fit_transform(df_list[0][columns])  # 无量纲化
    select_model.fit_transform(X, df_list[0]['label'])
    selected_index = select_model.get_support(indices=True)  # 返回被选择特征的索引
    new_columns = []
    for index in selected_index:  # 重构新的特征变量
        new_columns.append(columns[index])
    for i in range(len(df_list)):
        tem_df = df_list[i][new_columns]
        if i < 1:
            tem_df['label'] = df_list[i]['label']
        df_list[i] = tem_df
    return df_list


def year_judge(row):
    """
    年份判断
    :param row:
    :return:
    """
    if row > -8:
        row = int(row)
    else:
        return row
    if (row > 0) and (row < 1950):
        return 4
    elif (row >= 1950) and (row < 1960):
        return 5
    elif (row >= 1960) and (row < 1970):
        return 6
    elif (row >= 1970) and (row < 1980):
        return 7
    elif (row >= 1980) and (row < 1990):
        return 8
    elif (row >= 1990) and (row < 2000):
        return 9
    elif (row >= 2000) and (row < 2100):
        return 10
    else:
        return 0


def year_partition(df_list, year_type):
    """
    对年份标称型特征进行划分：50后 60后　70后　80后　90后 00后
    :param df_list: 训练集与验证集与测试集
    :param year_type: 年份标称型特征
    :return:
    """
    for i in range(len(df_list)):
        for col in year_type:
            df_list[i][col] = df_list[i][col].apply(year_judge)
    return df_list


def pca_decompose(data_mat, n):
    """
    主成分分析降维
    :param data_mat: 原始矩阵
    :return:
    """
    pca = decomposition.PCA(n_components=n)
    pca_data_mat = pca.fit_transform(data_mat)
    reduction = pca.explained_variance_ratio_ .sum()  # 查看降维效果
    print(pca.explained_variance_ratio_)
    print(reduction)
    print(pca_data_mat.shape)
    return pca_data_mat


def check_model(x, y):
    """
    拟合的模型
    :return:返回最佳模型
    """
    gbdt_model = GradientBoostingClassifier(
                            learning_rate=0.05,  # 学习率
                            n_estimators=10000,  # 迭代次数
                            min_samples_leaf=70,
                            max_depth=7,
                            subsample=0.85,
                            random_state=10,
                            min_samples_split=100,
                            )
    lr_model = lambda: LogisticRegression(
        penalty='l1',  # l1 & l2  小数据集用l1
        fit_intercept=True,  # 是否存在截距，默认存在
        max_iter=95,
        # Weights associated with classes. If not given, all classes are supposed to have weight one.
        class_weight={0: 0.9, 1: 0.1},
        solver='liblinear',
        C=10,

    )
    lr_pipe = Pipeline(steps=[
            ('ss', StandardScaler()),
            # transformer  # SGDClassifier对于特征的幅度非常敏感，也就是说，
            # 我们在把数据灌给它之前，应该先对特征做幅度调整，
            # 当然，用sklearn的StandardScaler可以很方便地完成这一点
            ('en', lr_model())  # estimator
        ])

    xgb_model = xgb.XGBClassifier(
        learning_rate=0.05,  # 学习率
        n_estimators=3800,  # 迭代次数
        max_depth=9,
        subsample=0.9,
        random_state=10,
        min_child_weight=1,
        colsample_bytree=0.8,
        seed=0,
        gamma=0.3,
        reg_alpha=1,
        reg_lambda=0.05
    )
    parm = {
            # GBDT调参
            # 'learning_rate': [0.05, 0.1],
            # 'n_estimators': range(800, 4000, 300),  # 迭代范围
            # 'max_features': range(2, 5, 1), # 最大特征数
            # 'max_depth': range(3, 14, 2),
            # 'min_samples_split': range(100, 801, 200),
            # 'max_features': range(3, 5, 1),
            # 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],

            #  LR调参
            # 'en__max_iter': range(10, 150, 5),
            # 'en__penalty': ('l1', 'l2'),
            # 'en__C': (0.01, 0.1, 1, 2.5, 5, 7.5, 10),

            # xgb调参
            # 'learning_rate': [0.05, 0.1],
            # 'n_estimators': range(800, 4000, 300),  # 迭代范围
            # 'reg_alpha': [0.05, 0.1, 1, 2, 3],
            # 'reg_lambda': [0.05, 0.1, 1, 2, 3],
            }  # 参数
    g_search = GridSearchCV(estimator=xgb_model,  # gbdt_model lr_pipe xgb_model
                            param_grid=parm,  # 参数
                            scoring='roc_auc',
                            cv=10,  # 10折交叉验证  一定程度上用来减少过拟合
                            iid=False,  # 独立同分布
                            n_jobs=-1,  # -1 means using all processors
                            verbose=1  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。
                            )
    g_search.fit(x, y)  # 运行网格搜索
    print(g_search.cv_results_['params'])  # 打印候选参数表
    print(g_search.cv_results_['mean_test_score'])  # 打印不同参数下的训练分数
    print(g_search.best_score_)  # 最佳分数
    print(g_search.best_params_)  # 最佳参数

    return g_search.best_estimator_  # 返回最佳模型


def save_model(model, model_name):
    """
    保存模型
    :param model:
    :param model_name:使用的模型名称
    :return:
    """
    print('Saving %s model.....' % model_name)
    with open('model/' + model_name + '_family_loss_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('Done!')


def get_model(model_name):
    """
    获得模型
    :param model_name:使用的模型名称
    :return:
    """
    print('Getting %s model.....' % model_name)
    with open('model/' + model_name + '_family_loss_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print('Done!')
    return model


def nominal_type_statistic(data_set, col):
    """
    标称型数据类型占比统计
    将标称型数据下的数据类型的占比进行统计，从大到小相加，当占比超过95％的时候，剩余小于5％合并成一类，
    并返回合并形成一个集合，返回。
    :param data_set:
    :param col: 统计的列名
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
    col_dict = dict(zip(col_list, col_rate_list))

    fina_data_list = list()  # 最终返回的数据类型列表
    threshold_rate = 0.85  # 阈值占比
    sum_rate = 0.0  # 累计占比
    name_rate = []  # 饼图的占比
    for key in col_dict.keys():
        if sum_rate >= threshold_rate:
            fina_data_list.append(key)  # 剩余的类别归并成一类
            name_rate.append(1 - sum_rate)
            break
        sum_rate += col_dict[key]
        name_rate.append(col_dict[key])
        fina_data_list.append(key)
    # 生成饼图
    # draw_pie_chart(fina_data_list, name_rate, col + ' pie chart')
    return fina_data_list


def draw_polygonal_line(x, y, title, color='bx-', x_label='X', y_label='Y'):
    """
    绘制折线图
    :param x: 横坐标
    :param y: 纵坐标
    :param title: 题目
    :param color: 颜色以及线条
    :param x_label: 横坐标标签名
    :param y_label: 纵坐标标签名
    :return:
    """
    plt.ion()  # 显示图片
    plt.pause(1)  # 显示秒数
    plt.plot(x, y, color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(image_to_save + title)
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


def draw_confuse_matrix(y_true, y_preb,  title='Confusion Matrix'):
    """
    画混淆矩阵
    :param y_true: 真实标签
    :param y_preb: 预测标签
    :param title: 表名
    :return:
    """
    plt.ion()  # 显示图片
    labels = list(set(y_true))
    cm = confusion_matrix(y_true, y_preb, labels=labels)
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.xlabel('True label')
    plt.ylabel('Predicted label ')
    for first_index in range(len(cm)):
        for second_index in range(len(cm[first_index])):
            plt.text(first_index, second_index, cm[first_index][second_index], bbox=dict(facecolor="r", alpha=0.65))
    plt.savefig(image_to_save + 'Confusion_Matrix')
    plt.pause(5)  # 显示秒数
    plt.close()


def draw_pr_curve(recall, precision, title='P-R Curve'):
    """
    画PR曲线
    :param recall: 召回率
    :param precision: 精确率
    :param title: 图片标题
    :return:
    """
    plt.ion()  # 显示图片
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall Rate')
    plt.ylabel('Precision Rate')
    plt.title(title)
    plt.savefig(image_to_save + 'PR_Curve')
    plt.pause(5)  # 显示秒数
    plt.close()


def sample_equilibrium(k, train_set):
    """
    样本均衡
    训练集的多数类分成ｋ等份，并对应与少数类合成训练子集
    :param k: k等份
    :param train_set:
    :return:
    """
    more_class_set = train_set[train_set['label'] == 0].copy()  # 多数类
    more_class_set.reset_index(inplace=True)  # 重新设置索引
    less_class_set = train_set[train_set['label'] == 1].copy()  # 少数类
    less_class_set.reset_index(inplace=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=9)
    # shuffle=False　每次划分结果都一样 True为随机索引 random_state=9:保持每次抽的索引都一样
    train_set_list = []  # 将分割后后的多数类集合分别和少数类合成一个新训练集，再将其装载在该列表
    for train_index, test_index in kf.split(more_class_set):
        # test_index只是索引
        print(test_index.shape[0] / less_class_set.shape[0])
        test = more_class_set.loc[test_index].copy()  # 取出该索引对应的数据
        sub_train_set = pd.concat([test, less_class_set], axis=0)  # 纵向链接(１是横向)  生成一个训练子集
        train_set_list.append(sub_train_set)
        print('---------------')

    return train_set_list


def single_model_train_and_predict(model_name):
    """
    对训练集和测试集进行特征工程处理
    将训练集分为0.8训练集 + 0.2验证集合
    单模训练与验证
    利用模型对测试集进行预测
    :param model_name:使用的模型名称
    对数据集进行标准化处理
    :return:
    """
    train_set = pd.read_csv(data_to_train_and_test + 'train_set.csv')
    test_set = pd.read_csv(data_to_train_and_test + 'test_set.csv')
    df_list = [train_set, test_set]  # 将训练集和测试集放入列表中

    df_list = year_partition(df_list, year_type)  # 将年份的分桶
    # 标称型缺失值统一为-30
    for df in df_list:
        df[nominal_type] = df[nominal_type].fillna(-30)
    family_loss_set = df_list[0][df_list[0]['label'] == 1].copy()  # 抽出label为１的个人流失数据

    for nt in nominal_type:  # 对特定的标称型特征进行占比统计
        fina_key_list = nominal_type_statistic(family_loss_set, nt)  # 得到最终的占比统计类别与剩余类别　
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
    for df in df_list:
        print(df.head(5))
    new_feature = new_nominal_type + new_numerical_type  # 新的特征列
    print(new_feature)

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

    # 建模
    new_train_set, new_valid_set = train_test_split(pca_train_set, test_size=0.2, stratify=pca_train_set['label'], random_state=100)
    new_test_set = pca_test_set

    """
    XGBoost 与 GBDT 简单加权融合，获取模型融合后的最终结果
    多模融合评估与预测
    """
    multiply_model_evaluate(new_valid_set, new_test_set, new_pridictors)
    """
    单模训练、评估与预测，获取选择的单模的最终结果
    """
    print('---------Process: single model train, assess and predict---------')
    # 训练
    """
    若要训练模型，则直接打开下面三行注释；
    """
    # x = new_train_set[new_pridictors]
    # y = new_train_set['label']
    # model = check_model(x, y)
    """
    若想要直接使用参赛者训练好的模型，则使用下行代码，注释上面三行代码！
    """
    model = get_model(model_name)

    #  验证
    x_valid = new_valid_set[new_pridictors]
    y_valid = new_valid_set['label']
    y_pred = model.predict(x_valid)
    y_pred_prob = model.predict_proba(x_valid)[:, 1]
    auc_score = roc_auc_score(y_valid, y_pred_prob)
    print('Accuracy: %.4g' % accuracy_score(y_valid.values, y_pred))
    print('Precision: %.4g' % precision_score(y_valid.values, y_pred))
    print('Recall: %.4g' % recall_score(y_valid.values, y_pred))
    print('F1: %.4g' % f1_score(y_valid.values, y_pred))
    print('AUC Score (Train): %.4g' % auc_score)

    # # 绘制roc曲线
    # positive_rate = 1 / (new_valid_set[new_valid_set['label'] == 1].shape[0])  # 正样本刻度
    # negative_rate = 1 / (new_valid_set[new_valid_set['label'] == 0].shape[0])  # 负样本刻度
    # temp_dict = {'label': y_valid, 'pred_prob': y_pred_prob}
    # temp = pd.DataFrame(temp_dict)
    # temp.sort_values('pred_prob', inplace=True, ascending=False)  # 基于预测概率倒序排序
    # draw_roc_line(positive_rate, negative_rate, list(temp['label']), auc_score)
    #
    # # 绘制混淆矩阵
    # draw_confuse_matrix(list(y_valid), list(y_pred))
    #
    # # 绘制P-R曲线
    # precision, recall, _ = precision_recall_curve(y_valid, y_pred_prob)
    # draw_pr_curve(recall, precision)

    # 测试
    test_x = new_test_set[new_pridictors]
    test_y_pred_prob = model.predict_proba(test_x)[:, 1]
    test_y_label = model.predict(test_x)
    new_test_set['pred'] = test_y_pred_prob
    new_test_set['label'] = test_y_label
    summit_file = new_test_set[['pid', 'fid', 'pred', 'label']].copy()
    summit_file.columns = ['pid', 'fid', 'pred', 'label']
    summit_file.to_csv('summit.csv', columns=['pid', 'fid', 'pred', 'label'], index=False)  # 不要索引, header=False 不要列头

    """
    获取最终结果 模型选择结果：0代表单模，１代表多模
    """
    get_fina_result(0)
    # # 保存模型
    save_model(model, model_name)
    print('Process:single model train, assess and predict Done!')


def multiply_model_evaluate(valid_set, test_set, features):
    """
    模型融合评估
    :param valid_set:验证集
    :param test_set:测试集
    :param features:特征
    :return:
    """
    print('---------Process: multiply model train, assess and predict---------')
    xgb_model = get_model('XGB')
    gbdt_model = get_model('GBDT')
    xgb_weight_list = []  # xgb权值列表
    gbdt_weight_list = []  # gbdt权值列表
    accuracy_score_list = []  # 准确率列表
    precision_score_list = []  # 精确率列表
    recall_score_list = []  # 召回率列表
    f1_score_list = []  # f1分数列表
    auc_score_list = []  # auc分数列表

    xgb_weight = 0.85
    while xgb_weight < 1.0:
        x_valid = valid_set[features]
        y_valid = valid_set['label']

        # 加权概率
        xgb_pred_prob = xgb_model.predict_proba(x_valid)[:, 1]
        gbdt_pred_prob = gbdt_model.predict_proba(x_valid)[:, 1]
        y_pred_prob = xgb_weight * xgb_pred_prob + (1-xgb_weight) * gbdt_pred_prob

        valid_set['pred'] = y_pred_prob  # 概率
        valid_set['predict_label'] = valid_set['pred'].apply(lambda x: 1 if x >= 0.5 else 0)  # 基于概率上标签
        y_pred = valid_set['predict_label'].values  # 预测的标签
        auc_score = roc_auc_score(y_valid, y_pred_prob)  # auc

        # 打印评估结果
        print('{xgb_weight: %s, gbdt_weight: %s}' % (str(xgb_weight), str(1-xgb_weight)))
        print(valid_set[valid_set['label'] == 1].shape[0])
        print('Accuracy: %.4g' % accuracy_score(y_valid.values, y_pred))
        print('Precision: %.4g' % precision_score(y_valid.values, y_pred))
        print('Recall: %.4g' % recall_score(y_valid.values, y_pred))
        print('F1: %.4g' % f1_score(y_valid.values, y_pred))
        print('AUC Score (Train): %.4g' % auc_score)
        print('--------------')
        xgb_weight_list.append(xgb_weight)
        gbdt_weight_list.append(1-xgb_weight)
        accuracy_score_list.append(accuracy_score(y_valid.values, y_pred))
        precision_score_list.append(precision_score(y_valid.values, y_pred))
        recall_score_list.append(recall_score(y_valid.values, y_pred))
        f1_score_list.append(f1_score(y_valid.values, y_pred))
        auc_score_list.append(auc_score)

        if xgb_weight == 0.85:  # 最佳权重
            # # 绘制roc曲线
            # positive_rate = 1 / (valid_set[valid_set['label'] == 1].shape[0])  # 正样本刻度
            # negative_rate = 1 / (valid_set[valid_set['label'] == 0].shape[0])  # 负样本刻度
            # temp_dict = {'label': y_valid, 'pred_prob': y_pred_prob}
            # temp = pd.DataFrame(temp_dict)
            # temp.sort_values('pred_prob', inplace=True, ascending=False)  # 基于预测概率倒序排序
            # draw_roc_line(positive_rate, negative_rate, list(temp['label']), auc_score)
            # # 绘制混淆矩阵
            # draw_confuse_matrix(list(y_valid), list(y_pred))
            #
            # # 绘制P-R曲线
            # precision, recall, _ = precision_recall_curve(y_valid, y_pred_prob)
            # draw_pr_curve(recall, precision)

            # 测试
            test_x = test_set[features]
            # 加权概率
            xgb_pred_prob = xgb_model.predict_proba(test_x)[:, 1]
            gbdt_pred_prob = gbdt_model.predict_proba(test_x)[:, 1]
            y_pred_prob = xgb_weight * xgb_pred_prob + (1 - xgb_weight) * gbdt_pred_prob
            test_set['pred'] = y_pred_prob  # 概率
            test_set['label'] = test_set['pred'].apply(lambda x: 1 if x >= 0.5 else 0)  # 基于概率上标签
            summit_file = test_set[['pid', 'fid', 'pred', 'label']].copy()
            summit_file.columns = ['pid', 'fid', 'pred', 'label']
            summit_file.to_csv('summit.csv', columns=['pid', 'fid', 'pred', 'label'], index=False)
            get_fina_result(1)  # 获取最终结果 模型选择结果：0代表单模，１代表多模
            break
        xgb_weight += 0.05  # 权值增加

    # 将评估结果形成csv文件
    show_table_dict = {
        'xgb_weight': xgb_weight_list,
        'gbdt_weight': gbdt_weight_list,
        'accuracy': auc_score_list,
        'precision_rate': precision_score_list,
        'recall_rate': recall_score_list,
        'F1_score': f1_score_list,
        'AUC_score': auc_score_list,
    }
    show_table_df = pd.DataFrame(show_table_dict)
    show_table_df.to_csv('show_table.csv', index=False)  # 不要索引
    print('Process: multiply model train, assess and predict Done!')


def get_fina_result(choice):
    """
    :param choice:　模型选择结果：0代表单模，１代表多模
    生成最终流失的1000个fid
    :return:
    """
    result = pd.read_csv('summit.csv')
    result = result[['fid', 'pred']].copy()
    result['count'] = 1
    temp = result.groupby(['fid'], as_index=False).count()  # 统计每个fid出现的次数
    res = (result.groupby(['fid'], as_index=False)['pred'].sum()).copy()  # 基于fid求和流失概率
    res['fid_count'] = temp['count']
    res['final_probability'] = res.apply(lambda x: x['pred'] / x['fid_count'], axis=1)  # 二者相除得出流失的概率
    res = (res.sort_values('final_probability', ascending=False)).copy()  # 降序排序
    if choice:
        res.to_csv('final_disappear_fid_multiply_model.csv', columns=['fid', 'final_probability'], index=False)
    else:
        res.to_csv('final_disappear_fid_single_model_gbdt.csv', columns=['fid', 'final_probability'], index=False)


if __name__ == '__main__':
    single_model_train_and_predict('GBDT')  # GBDT训练、评估与预测
    # single_model_train_and_predict('XGB')  # XGB训练、评估与预测

    # df_draw = pd.read_csv('show_table.csv')
    # # df_draw.plot(kind='bar')  # 柱状图
    # # df_draw['AUC_score'].plot(kind='pie')  # 饼图
    # df_draw.plot()  # 默认折线
    # # df_draw.plot(kind='bar', stacked=True)  # 堆叠柱状图
    # plt.ion()
    # # plt.figure(1, (20, 20))
    # plt.pause(5)  # 显示秒数
    # plt.savefig('show_table')
    # plt.close()



