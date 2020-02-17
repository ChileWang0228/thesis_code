import os
import json
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("/home/ubuntu/data/complete_code/CCF_ner")  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config
from train_fine_tune import  get_P_R_F
config = Config()

def vote(y_list):
    y_voted_list = []
    print('Getting Result.....')
    for key in tqdm(y_list.keys()):
        pred_key = np.concatenate(y_list[key])  # 3维
        j = 0
        temp_list = []
        for i in range(config.batch_size):
            temp = []
            while True:
                try:
                    temp.append(pred_key[j])
                    j += config.batch_size
                except:
                    j = 0
                    j += i + 1
                    break

            temp_T = np.array(temp).T  # 转置
            pred = []
            for line in temp_T:
                pred.append(np.argmax(np.bincount(line)))  # 找出列表中出现次数最多的值
            temp_list.append(pred)
        y_voted_list.append(temp_list)
    return y_voted_list

def vote_ensemble(path, dataset, output_path, remove_list):
    single_model_list = [x for x in os.listdir(path) if dataset + '_result_detail' in x]
    print('ensemble from file: ')
    for file_name in single_model_list:
        print(file_name)

    pred_list = OrderedDict()
    true_list = OrderedDict()
    for index, file in enumerate(single_model_list):
        if file not in remove_list:  # 预测所有模型
            print(index)
            print('Text File: ', file)
            break
    print('Ensembling.....')
    for index, file in enumerate(single_model_list):
        if file in remove_list:
            continue
        print('Ensemble file:', file)
        with open(path + file) as f:
            for i, line in tqdm(enumerate(f.readlines())):
                item = json.loads(line)

                if i not in pred_list:
                    pred_list[i] = []
                    true_list[i] = []
                pred_list[i].append(item['pred'])
                true_list[i].append(item['true'])


    y_pred_list = vote(pred_list)
    y_true_list = vote(true_list)
    y_pred_list = np.concatenate(y_pred_list)
    y_true_list = np.concatenate(y_true_list)

    print(y_true_list)
    print(y_pred_list)
    precision, recall, f1 = get_P_R_F(y_pred_list, y_true_list)

    print('precision: {}, recall {}, f1 {}'.format(precision, recall, f1))



if __name__ == '__main__':

    remove_list = [
        # 'test_result_detail_model_0.6774_0.7129-6235.txt',  # 原生bert768 LSTM 256  F1=0.48 315kb
        # 'test_result_detail_model_0.6841_0.6867-4988.txt',  # 动态融合 LSTM=128  F1=0.496 313kb
        # 'test_result_detail_model_0.6472_0.6642-2494.txt',  # BERT + IDCNN Bert最后一层向量 512 298kb


                   ]
    # 测试集
    # score_average_ensemble(config.ensemble_source_file, 'test', config.ensemble_result_file, remove_list)
    vote_ensemble(config.ensemble_source_file, 'test', config.ensemble_result_file, remove_list)

    # 验证集
    # vote_ensemble(config.ensemble_source_file, 'dev', config.ensemble_result_file, remove_list)
    # score_average_ensemble(config.ensemble_source_file, 'dev', config.ensemble_result_file, remove_list)

