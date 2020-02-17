#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行环境：Anaconda python 3.7.2
@Created on 2019-3-12 21:19
@Author:ChileWang
@algorithm：
暂时用不到的函数
"""
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# def select_best_k_for_kmeans(more_class_set):
#     """
#     找到最佳的k均值分类
#     :param more_class_set:多数类样本
#     :return:
#     """
#     print(more_class_set[numerical_type].head(5))
#     k_list = range(1, 10)
#     mean_num_list = []  # 装载每个数值型特征原本的平均数
#     std_num_list = []  # 装载每个数值型特征原本的标准差
#     for num_t in numerical_type:
#         mean_num = more_class_set[num_t].mean()  # 该列平均值
#         mean_num_list.append(mean_num)
#         std_num = more_class_set[num_t].std()  # 该列标准差
#         std_num_list.append(std_num)
#         more_class_set[num_t] = more_class_set[num_t].apply(lambda x: 1.0 * (x-mean_num)/std_num)  # 数值型每一列进行数据标准化
#
#     mean_distortions = []  # 计算所有点与对应聚类中心的距离的平方和的均值
#     for k in k_list:
#         kmeans = KMeans(n_clusters=k, n_jobs=-1)
#         kmeans.fit(more_class_set)
#         # 解释此处代码
#         # cdist:Computes distance between each pair of the two collections of inputs.
#         # 理解为计算某个与其所属类聚中心的欧式距离
#         # 最终是计算所有点与对应中心的距离的平方和的均值
#         mean_distortions.append(sum(np.min(cdist(more_class_set, kmeans.cluster_centers_, 'euclidean'),
#                                            axis=1))/more_class_set.shape[0])
#     draw_polygonal_line(k_list, mean_distortions, title='Selecting k with the Elbow Method',
#                         color='bx-', x_label='K', y_label='Average Dispersion')
#
#     for i in range(len(numerical_type)):
#         num_t = numerical_type[i]
#         mean_num = mean_num_list[i]  # 该列原本平均值
#         std_num = std_num_list[i]  # 该列原本标准差
#         more_class_set[num_t] = more_class_set[num_t].apply(lambda x: round(x * std_num + mean_num, 1))  # 复原数值型每一行数据
#     more_class_set[numerical_type] = more_class_set[numerical_type].replace(-0.0, 0.0)  # replace函数是右边替换左边
#     print(more_class_set[numerical_type].head(5))

# def sample_equilibrium(k, train_set, feature):
#     """
#     :param k: 最合适的累簇数量
#     :param train_set: 训练集
#     :param feature: 用来聚类的特征
#     :return: 均衡的样本
#     """
#     mean_num_list = []  # 装载每个数值型特征原本的平均数
#     std_num_list = []  # 装载每个数值型特征原本的标准差
#     more_class_set = train_set[train_set['label'] == 0].copy()  # 多数类
#     more_class_set.reset_index(inplace=True)  # 重新设置索引
#     less_class_set = train_set[train_set['label'] == 1].copy()  # 少数类
#     less_class_set.reset_index(inplace=True)
#
#     # 对多数类数值型每一列进行数据标准化
#     for num_t in numerical_type:
#         mean_num = more_class_set[num_t].mean()  # 该列平均值
#         mean_num_list.append(mean_num)
#         std_num = more_class_set[num_t].std()  # 该列标准差
#         std_num_list.append(std_num)
#         more_class_set[num_t] = more_class_set[num_t].apply(lambda x: 1.0 * (x - mean_num) / std_num)
#
#     # k-means聚类
#     print('---------K-means-------------')
#     model = KMeans(n_clusters=k, n_jobs=-1, random_state=9)
#     model.fit(more_class_set[feature])
#     r1 = pd.Series(model.labels_).value_counts()  # 统计每个类别的数目
#     r2 = pd.DataFrame(model.cluster_centers_)  # 找出每个类别的聚类中心
#     r = pd.concat([r2, r1], axis=1)  # 横向链接(0是纵向),得到聚类中心对应的类别下的数目
#     # print(r)
#
#     kmeans_df = []  # 多数类聚类后的数据集列表
#     for i in range(k):
#         index = np.where(model.labels_ == i)  # 取出属于该簇类的所有索引
#
#         kmeans_df.append(more_class_set.loc[index].copy())
#
#     # 对多数类数值型进行复原
#     for c_df in kmeans_df:
#         for i in range(len(mean_num_list)):
#             mean_num = mean_num_list[i]
#             std_num = std_num_list[i]
#             num_t = numerical_type[i]
#             c_df[num_t] = c_df[num_t].apply(lambda x: round(x * std_num + mean_num, 1))  # 复原数值型每一行数据
#         c_df[numerical_type] = c_df[numerical_type].replace(-0.0, 0.0)  # replace函数是右边替换左边
#
#     train_set_list = []  # 将聚类后的多数类集合分别和少数类合成一个新训练集，再将其装载在该列表
#     for i in range(k):
#         print(kmeans_df[i].shape[0] / less_class_set.shape[0])
#         print('---------------')
#         train_set = pd.concat([kmeans_df[i], less_class_set], axis=0)  # 纵向链接(１是横向)
#         train_set_list.append(train_set)
#
#     return train_set_list

# def multiply_model_evaluate(valid_set, test_set, features, model_name_list):
#     """
#     模型融合评估
#     :param valid_set: 验证集
#     :param test_set: 测试集
#     :param features: 特征
#     :param model_name_list: 模型名字
#     :return:
#     """
#     accuracy_list = []  # 子模型准确度列表
#     precision_list = []  # 子模型精度列表
#     recall_list = []  # 子模型召回度列表
#     auc_score_list = []  # 子模型auc列表
#     f1_score_list = []  # 子模型f1分数列表
#     k = len(model_name_list)  # 子模型数目
#     for model_name in model_name_list:
#         model = get_model(model_name)
#         # 验证
#         x_valid = valid_set[features]
#         y_valid = valid_set['label']
#         y_pred = model.predict(x_valid)  # 标签
#         y_pred_prob = model.predict_proba(x_valid)[:, 1]  # 概率
#         auc_score = roc_auc_score(y_valid, y_pred_prob)
#         print('Accuracy: %.4g' % accuracy_score(y_valid.values, y_pred))
#         print('Precision: %.4g' % precision_score(y_valid.values, y_pred))
#         print('Recall: %.4g' % recall_score(y_valid.values, y_pred))
#         print('F1: %.4g' % f1_score(y_valid.values, y_pred))
#         print('AUC Score (Train): %.4g' % auc_score)
#         auc_score_list.append(auc_score)
#         accuracy_list.append(accuracy_score(y_valid.values, y_pred))
#         precision_list.append(precision_score(y_valid.values, y_pred))
#         recall_list.append(recall_score(y_valid.values, y_pred))
#         f1_score_list.append(f1_score(y_valid.values, y_pred))
#
#     # 计算平均值
#     auc_score_mean = np.mean(auc_score_list)
#     auc_score_list.append(auc_score_mean)
#
#     accuracy_mean = np.mean(accuracy_list)
#     accuracy_list.append(accuracy_mean)
#
#     precision_mean = np.mean(precision_list)
#     precision_list.append(precision_mean)
#
#     recall_mean = np.mean(recall_list)
#     recall_list.append(recall_mean)
#
#     f1_score_mean = np.mean(f1_score_list)
#     f1_score_list.append(f1_score_mean)
#     # 生成csv文件
#     sample_name_list = ['S' + str(i + 1) for i in range(k)]  # 样本名称
#     sample_name_list.append('average')  # 平均数
#     show_table = {
#         'Sample': sample_name_list,
#         'auc_score': auc_score_list,
#         'accuracy': accuracy_list,
#         'precision': precision_list,
#         'recall': recall_list,
#         'F1_score': f1_score_list
#     }
#     show_table_df = pd.DataFrame(show_table)  # 形成Dataframe
#     show_table_df.to_csv('show_table.csv', index=False)  # 不要索引
#
#     # 测试
#     test_set['pred'] = 0
#     for model_name in model_name_list:
#         model = get_model(model_name)
#         test_x = test_set[features]
#         test_y_pred_prob = model.predict_proba(test_x)[:, 1]
#         # test_y_label = model.predict(test_x)
#         test_set['pred'] += test_y_pred_prob
#         # new_test_set['label'] = test_y_label
#         print(test_set['pred'].head(5))
#     test_set['pred'] = test_set['pred'].apply(lambda x: x / k)  # 求概率平均值
#     test_set['label'] = test_set['pred'].apply(lambda x: 1 if x >= 0.5 else 0)  # 根据概率上标签
#     print(test_set['pred'].head(5))
#     print(test_set[test_set['label'] == 1].shape[0])
#     summit_file = test_set[['pid', 'fid', 'pred', 'label']].copy()
#     summit_file.columns = ['pid', 'fid', 'pred', 'label']
#     summit_file.to_csv('summit.csv', columns=['pid', 'fid', 'pred', 'label'], index=False)  # 不要索引, header=False
#
#
# def multiply_model_merging():
#     """
#     多模融合
#     对训练集和测试集进行特征工程处理
#     取训练集的80%做训练集，剩下20%做验证集
#     训练集分为多数类(0)和少数类(1)
#     将多数类分为K份(多数类和少数类的比值)，分别与少数类形成训练子集，分别训练模型
#     得到K个模型，分别对验证集进行验证，统计精确率, 召回率，F1, AUC
#     利用投票原则统计最终的精确率, 召回率，F1, AUC
#     :return:
#     """
#     train_set = pd.read_csv(data_to_train_and_test + 'train_set.csv')
#     test_set = pd.read_csv(data_to_train_and_test + 'test_set.csv')
#     df_list = [train_set, test_set]  # 将训练集和测试集放入列表中
#
#     df_list = year_partition(df_list, year_type)  # 将年份的分桶
#     # 标称型缺失值统一为-30
#     for df in df_list:
#         df[nominal_type] = df[nominal_type].fillna(-30)
#     family_loss_set = df_list[0][df_list[0]['label'] == 1].copy()  # 抽出label为１的个人流失数据
#
#     for nt in nominal_type:  # 对特定的标称型特征进行占比统计
#         fina_key_list = nominal_type_statistic(family_loss_set, nt)  # 得到最终的占比统计类别与剩余类别　
#         # 对训练集和测试集合相应的特征进行新的分类, 若类在于fina_key_list当中，则继续应用，否则将以得到的剩余类别将其分类
#         df_list[0][nt] = df_list[0][nt].apply(lambda x: x if x in fina_key_list else fina_key_list[-1])
#         df_list[1][nt] = df_list[1][nt].apply(lambda x: x if x in fina_key_list else fina_key_list[-1])
#
#     # 数值型特征缺失值补充
#     df_list = medium_fill(df_list, numerical_type)
#
#     # 样本均衡
#     # 将训练集分为0.8训练集　＋　0.2验证集
#     total_train_set, total_valid_set = train_test_split(df_list[0], test_size=0.2, stratify=df_list[0]['label'], random_state=100)
#     more_class_set = total_train_set[total_train_set['label'] == 0].copy()  # 多数类样本
#     less_class_set = total_train_set[total_train_set['label'] == 1].copy()  # 少数类样本
#     imbalance_rate = more_class_set.shape[0] / less_class_set.shape[0]  # 不平衡度
#     k = round(imbalance_rate)  # 四舍五入 聚类ｋ值
#     train_set_list = sample_equilibrium(k, total_train_set)  # train_set_list:K个训练子集
#
#     new_train_and_test_list = []  # 装载训练子集与测试集合
#     new_valid_and_test_list = [total_valid_set, df_list[1]]  # 装载总的验证集与测试集合
#     for train_set in train_set_list:
#         new_train_and_test_list.append([train_set, df_list[1]])
#
#     # 对new_train_and_test_list
#     model_name_list = ['gbdt_' + str(i) for i in range(k)]  # 模型名列表
#     name_index = 0  # 模型名字索引
#     for df_list in new_train_and_test_list:
#         # 特征选择
#         df_list = var_delete(df_list, nominal_type, numerical_type)  # 方差选择法
#         df_list = select_best_chi2(df_list)  # 卡方验证选择法
#
#         # 标称型独热编码
#         columns = df_list[1].columns.values.tolist()  # 测试集不包含label
#         new_nominal_type = list(set(columns) & set(nominal_type))  # 新的标称型列
#         new_numerical_type = list(set(columns) & set(numerical_type))  # 新的数值型列
#         df_list, new_nominal_type = one_hot_encode(df_list, new_nominal_type)  # 独热编码
#         new_feature = new_nominal_type + new_numerical_type  # 新的特征列
#
#         # PCA降维
#         n_components = 5  # pca的维度
#         new_pridictors = ['a' + str(i) for i in range(n_components)]  # 降维后的新特征
#         pca_train_set = pd.DataFrame(pca_decompose(df_list[0][new_feature].values, n_components), columns=new_pridictors)  # DF转矩阵
#         # 将相应的列补上
#         pca_train_set['label'] = list(df_list[0]['label'])
#
#         # 建模
#         new_train_set = pca_train_set
#         # 训练
#         x = new_train_set[new_pridictors]
#         y = new_train_set['label']
#         model = check_model(x, y)
#
#         # 保存模型
#         save_model(model, model_name_list[name_index])
#         name_index += 1
#
#     # 对new_valid_and_test_list
#     # 特征选择
#     new_valid_and_test_list = var_delete(new_valid_and_test_list, nominal_type, numerical_type)  # 方差选择法
#     new_valid_and_test_list = select_best_chi2(new_valid_and_test_list)  # 卡方验证选择法
#
#     # 标称型独热编码
#     columns = new_valid_and_test_list[1].columns.values.tolist()  # 测试集不包含label
#     new_nominal_type = list(set(columns) & set(nominal_type))  # 新的标称型列
#     new_numerical_type = list(set(columns) & set(numerical_type))  # 新的数值型列
#     new_valid_and_test_list, new_nominal_type = one_hot_encode(new_valid_and_test_list, new_nominal_type)  # 独热编码
#     new_feature = new_nominal_type + new_numerical_type  # 新的特征列
#
#     # PCA降维
#     n_components = 5  # pca的维度
#     new_pridictors = ['a' + str(i) for i in range(n_components)]  # 降维后的新特征
#     pca_valid_set = pd.DataFrame(pca_decompose(new_valid_and_test_list[0][new_feature].values, n_components),
#                                  columns=new_pridictors)  # DF转矩阵
#     pca_test_set = pd.DataFrame(pca_decompose(new_valid_and_test_list[1][new_feature].values, n_components),
#                                 columns=new_pridictors)  # DF转矩阵
#     # 将相应的id补上
#     pca_valid_set['label'] = list(new_valid_and_test_list[0]['label'])
#     pca_test_set['pid'] = test_set['pid']
#     pca_test_set['fid'] = test_set['fid']
#
#     multiply_model_evaluate(pca_valid_set, pca_test_set, new_pridictors, model_name_list)


# ---------------test--------------
# numerical_money_type = ['qk601', 'fe601', 'fe802', 'fe903', 'ff2', 'indinc', 'land_asset', 'total_asset',
#                         'expense', 'fproperty']  # 数值型金钱特征
# numerical_income_type = ['qk601', 'fe601', 'fe802', 'fe903', 'ff2', 'indinc', 'land_asset', 'total_asset',
#                          'fproperty']  # 数值型收入特征
# for df in df_list:
#     for col in numerical_money_type:
#         df[col] = df[col].apply(lambda x: 0 if x == -8 else x)  # 将0替代-8
#
# # 数值型特征缺失值用每一列的中位数去填充
# df_list = medium_fill(df_list, numerical_type)
#
# for df in df_list:
#     df['total_funds'] = 0
#     for col in numerical_income_type:
#         df['total_funds'] += df[col]
#     df['total_funds'] -= df['expense']  # 减掉自己支出等于资金总和
# numerical_type.append('total_funds')
# print(numerical_type)
# print(df_list[0]['total_funds'].head(5))
# print(df_list[1]['total_funds'].head(5))
# --------------------------------------