import pandas as pd
import codecs
import re
import json
import sys
sys.path.append("/home/wangzhili/chilewang/Br_CCF")  # 添加项目根路径，避免在服务器上调用代码时找不到上一级目录的模块
from config import Config
from bert import tokenization
import numpy as np


"""
按照标点符号切割的预处理数据
"""
config = Config()
len_treshold = config.sequence_length - 2

data_dir = config.new_data_process_quarter_final
print(data_dir)
vocab_file = config.vocab_file
do_lower_case =False
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
# 原始数据集
train_df = pd.read_csv('/data/wangzhili/BR_NER/clean_csv_data/train.csv', encoding='utf-8')
dev_df = pd.read_csv('/data/wangzhili/BR_NER/clean_csv_data/dev.csv', encoding='utf-8')
test_df = pd.read_csv('/data/wangzhili/BR_NER/clean_csv_data/test.csv', encoding='utf-8')
print(train_df.shape)
print(dev_df.shape)
print(test_df.shape)


def token_label_train(text_list, label_list):
    """
    对文本的单词进行token，以便被字典所识别
    :param text_list:
    :param label_list:
    :return:
    """
    new_text_list = []
    new_label_list = []
    O_count = 0
    for i, text in enumerate(text_list):
        temp_list = text.split(' ')
        l_list = label_list[i].split(' ')

        if len(set(l_list)) == 1 and 'O' in set(l_list):
            O_count += 1
            continue
        temp_text_list = []
        temp_label_list = []
        j = 0
        for t in temp_list:
            temp = tokenizer.tokenize(t)  # 对每个单词tokenize
            flag = True
            for sub_t in temp:
                temp_text_list.append(sub_t)
                if flag:
                    temp_label_list.append(l_list[j])
                    j += 1
                    flag = False
                else:
                    temp_label_list.append('X')  # 多余的token用"X"表示
        new_text_list.append(" ".join(temp_text_list))
        new_label_list.append(" ".join(temp_label_list))
        # if  i < 2:
        #     print(text)
        #     print(label_list[i])
        #     print()
        #     print(new_text_list[i])
        #     print(new_label_list[i])
        #     print()
    print('abnormal sample:',O_count)
    return new_text_list, new_label_list

def token_label_dev(text_list, label_list):
    """
    对文本的单词进行token，以便被字典所识别
    :param text_list:
    :param label_list:
    :return:
    """
    new_text_list = []
    new_label_list = []
    for i, text in enumerate(text_list):
        temp_list = text.split(' ')
        l_list = label_list[i].split(' ')


        temp_text_list = []
        temp_label_list = []
        j = 0
        for t in temp_list:
            temp = tokenizer.tokenize(t)  # 对每个单词tokenize
            flag = True
            for sub_t in temp:
                temp_text_list.append(sub_t)
                if flag:
                    temp_label_list.append(l_list[j])
                    j += 1
                    flag = False
                else:
                    temp_label_list.append('X')  # 多余的token用"X"表示
        new_text_list.append(" ".join(temp_text_list))
        new_label_list.append(" ".join(temp_label_list))

    return new_text_list, new_label_list

print('Train')
train_text_list = train_df['text'].tolist()
train_label_list = train_df['label'].tolist()
train_text_list, train_label_list = token_label_train(train_text_list, train_label_list)
print('normal sample:', len(train_text_list))
# print(len(train_label_list))
print()
i = 0
j = 0
all = []
for text in train_text_list:
    temp_list = text.split(' ')
    temp = []
    for t in temp_list:
        temp += tokenizer.tokenize(t)
    all.append(len(temp))
    if len(temp) > 256:
        i += 1
    if len(temp) > 512:
        j += 1
        # print(text)
        # print()
print(i, j)
print(np.mean(all))
print(np.median(all))

print('Dev')
dev_text_list = dev_df['text'].tolist()
dev_label_list = dev_df['label'].tolist()
dev_text_list, dev_label_list = token_label_dev(dev_text_list, dev_label_list)
print('normal sample:', len(dev_text_list))
print()

print('Test')
test_text_list = test_df['text'].tolist()
test_label_list = test_df['label'].tolist()
test_text_list, test_label_list = token_label_dev(test_text_list, test_label_list)
print('normal sample:', len(test_text_list))
print()

def _cut(sentence):
    """
    将一段文本切分成多个句子
    :param sentence:
    :return:
    """
    new_sentence = []
    sen = []
    sentence = sentence.split(' ')
    for i in sentence:
        if i in [';'] and len(sen) != 0:
            sen.append(i)
            new_sentence.append(" ".join(sen))
            sen = []
            continue
        sen.append(i)
    new_sentence.append(" ".join(sen))  # 加上最后一句
    return new_sentence


# 数据切分
def cut_data(text_list, label_list):
    cut_text_list = []
    cut_label_list = []
    for i, text in enumerate(text_list):
        temp_cut_text_list = []
        temp_cut_label_list = []
        text_agg = []
        label_agg = []
        if len(text.split(' ')) < len_treshold:  # 小于最大切割值。
            temp_cut_text_list.append(text)
            temp_cut_label_list.append(label_list[i])
        else:
            l_list = label_list[i].split(' ')
            index = 0  # 记录当前切分的索引
            sentence_list = _cut(text)  # 一条数据被切分成多句话
            for sentence in sentence_list:
                if len(text_agg) + len(sentence.split(' ')) < len_treshold:
                    text_agg += sentence.split(' ')
                    label_agg += l_list[index: index + len(sentence.split(' '))]
                else:
                    temp_cut_text_list.append(" ".join(text_agg))
                    temp_cut_label_list.append(" ".join(label_agg))
                    text_agg = sentence.split(' ')
                    label_agg = l_list[index: index + len(sentence.split(' '))]
                index += len(sentence.split(' '))
            temp_cut_text_list.append(" ".join(text_agg))  # 加上最后一个句子
            temp_cut_label_list.append(" ".join(label_agg))

        # all = 0
        # all_l = 0
        # print(text.split(' '))
        # print([label_list[i]])
        # print(temp_cut_label_list)
        # print()
        # for i, t in enumerate(temp_cut_text_list):
        #     print(t.split(' '))
        #     print(temp_cut_label_list[i].split(' '))
        #     print(len(t.split(' ')))
        #     print(len(temp_cut_label_list[i].split(' ')))
        #     all += len(t.split(' '))
        #     all_l += len(temp_cut_label_list[i].split(' '))
        #     print()
        # print(all, all_l)



        cut_text_list += temp_cut_text_list
        cut_label_list += temp_cut_label_list

    return cut_text_list, cut_label_list



# for i, text in enumerate(train_text_list):
#     if len(text.split(' ')) > 256:
#         print(i)
#         print(text)
#         print(train_label_list[i])
#
# test_text = [train_text_list[77], train_text_list[90], train_text_list[91]]
# test_label = [train_label_list[77], train_label_list[90], train_label_list[91]]
train_text_list, train_label_list = cut_data(train_text_list, train_label_list)
dev_text_list, dev_label_list = cut_data(dev_text_list, dev_label_list)
test_text_list, test_label_list = cut_data(test_text_list, test_label_list)

print(train_text_list[: 10])
print(train_label_list[: 10])
# 构造训练集、验证集与测试集
with codecs.open(data_dir + 'train.txt', 'w', encoding='utf-8') as up:
    for i, text in enumerate(train_text_list):
        for c1, c2 in zip(text.split(' '), train_label_list[i].split(' ')):
            if len(c1) > 0:
                up.write('{0} {1}\n'.format(c1, c2[0]))
        up.write('\n')

with codecs.open(data_dir + 'dev.txt', 'w', encoding='utf-8') as up:
    for i, text in enumerate(dev_text_list):
        for c1, c2 in zip(text.split(' '), dev_label_list[i].split(' ')):
            if len(c1) > 0:
                up.write('{0} {1}\n'.format(c1, c2[0]))
        up.write('\n')

with codecs.open(data_dir + 'test.txt', 'w', encoding='utf-8') as up:
    for i, text in enumerate(test_text_list):
        for c1, c2 in zip(text.split(' '), test_label_list[i].split(' ')):

            if len(c1) > 0:
                up.write('{0} {1}\n'.format(c1, c2[0]))
        up.write('\n')


