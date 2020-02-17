import os
import time
import json
import tqdm
from config import Config
from tensorflow.contrib.crf import viterbi_decode
import tensorflow as tf
from utils import DataIterator
from optimization import create_optimizer
import numpy as np
from bert import tokenization
from model import Model
"""
# max len = 128



# new f1
GPU3: 原生 + bilstm=128    f1 0.9367708507777809 , p 0.9259000001 , r 0.9479000001
GPU4: 动态 + bilstm=128    f1 0.9187214364438824 , p 0.9030000001 , r 0.9350000001000001 


GPU5: 动态 + idcnn lstm=128   
GPU6: 原生 + idcnn lstm=128    
"""

gpu_id = 6
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

result_data_dir = Config().new_data_process_quarter_final
print('GPU ID: ', str(gpu_id))
print('Model Type: ', Config().model_type)
print('Fine Tune Learning Rate: ', Config().embed_learning_rate)
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('bilstm embedding ', Config().lstm_dim)
print('use original bert ', Config().use_origin_bert)
print('batch size ', Config().batch_size)


def train(train_iter, test_iter, config):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            model = Model(config)  # config.sequence_length,

            global_step = tf.Variable(0, name='step', trainable=False)
            learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.decay_step,
                                                       config.decay_rate, staircase=True)

            normal_optimizer = tf.train.AdamOptimizer(learning_rate)

            all_variables = graph.get_collection('trainable_variables')
            word2vec_var_list = [x for x in all_variables if 'bert' in x.name]
            normal_var_list = [x for x in all_variables if 'bert' not in x.name]
            print('bert train variable num: {}'.format(len(word2vec_var_list)))
            print('normal train variable num: {}'.format(len(normal_var_list)))
            normal_op = normal_optimizer.minimize(model.loss, global_step=global_step, var_list=normal_var_list)
            num_batch = int(train_iter.num_records / config.batch_size * config.train_epoch)
            embed_step = tf.Variable(0, name='step', trainable=False)
            if word2vec_var_list:  # 对bert微调
                print('word2vec trainable!!')
                word2vec_op, embed_learning_rate, embed_step = create_optimizer(
                    model.loss, config.embed_learning_rate, num_train_steps=num_batch,
                    num_warmup_steps=int(num_batch * 0.05) , use_tpu=False ,  variable_list=word2vec_var_list
                )

                train_op = tf.group(normal_op, word2vec_op)
            else:
                train_op = normal_op

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(
                os.path.join(config.model_dir, "runs_" + str(gpu_id), timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(out_dir + '/' + 'config.json', 'w', encoding='utf-8') as file:
                json.dump(config.__dict__, file)
            print("Writing to {}\n".format(out_dir))

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)
            if config.continue_training:
                print('recover from: {}'.format(config.checkpoint_path))
                saver.restore(session, config.checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())
            cum_step = 0
            for i in range(config.train_epoch):
                for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length in tqdm.tqdm(
                        train_iter):

                    feed_dict = {
                        model.input_x_word: input_ids_list,
                        model.input_mask: input_mask_list,
                        model.input_relation: label_ids_list,
                        model.input_x_len: seq_length,

                        model.keep_prob: config.keep_prob,
                        model.is_training: True,
                    }

                    _, step, _, loss, lr = session.run(
                            fetches=[train_op,
                                     global_step,
                                     embed_step,
                                     model.loss,
                                     learning_rate
                                     ],
                            feed_dict=feed_dict)


                    if cum_step % 10 == 0:
                        format_str = 'step {}, loss {:.4f} lr {:.5f}'
                        print(
                            format_str.format(
                                step, loss, lr)
                        )
                    cum_step += 1

                P, R = set_test(model, test_iter, session)

                print('dev set : step_{},precision_{},recall_{}'.format(cum_step, P, R))
                saver.save(session, os.path.join(out_dir, 'model_{:.4f}_{:.4f}'.format(P, R)),
                           global_step=step)


def decode(logits, lengths, matrix):
    """
    :param logits: [batch_size, num_steps, num_tags]float32, logits
    :param lengths: [batch_size]int32, real length of each sequence
    :param matrix: transaction matrix for inference
    :return:
    """
    # inference final labels usa viterbi Algorithm
    paths = []
    small = -1000.0
    start = np.asarray([[small] * Config().relation_num + [0]])
    # print('length:', lengths)
    for score, length in zip(logits, lengths):
        score = score[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)
        path, _ = viterbi_decode(logits, matrix)

        paths.append(path[1:])

    return paths


def get_P_R_F(y_pred_list, y_true_list):
    TP = 0
    FP = 0
    FN = 0
    for i, y_true_label in enumerate(y_true_list):
        current_TP = 0
        y_pred_label = y_pred_list[i]
        if 2 not in y_true_label:  #  全“0”标签
            if 2 not in y_pred_label:  # 不含有“B”标签
                current_TP += 1
            else:
                pred_set = set(y_pred_label)
                pred_dict = {}
                for item in pred_set:
                    pred_dict.update({item: y_pred_label.count(item)})  # 统计列表中每个元素出现的次数
                FP += pred_dict[2]  # 出现多少个“B”标签，FP就加几次
        else:  # 含有“B”标签
            entities_pos = []  # 记录每一个真实标签的Begin 和 End位置
            for i, value in enumerate(y_true_label):
                if value == 2:
                    begin = i
                    end = begin + 1
                    sub_v = y_true_label[end]
                    while sub_v in [3, 4]:  # 当标签是“X”或者"I"，循环查找end位置， 直到碰到别的标签，退出循环
                        end += 1
                        sub_v = y_true_label[end]
                    entities_pos.append([begin, end])
            for pos in entities_pos:
                start = pos[0]
                end = pos[1]
                y_pred = y_pred_label[start: end]
                y_true = y_true_label[start: end]
                flag = False

                if  len(y_pred)==0 or 2 != y_pred[0] or (2 == y_pred[0] and 1 in y_pred):
                    flag = True
                if flag:  # 误识别
                    FP += 1
                else:
                    current_TP += 1
            FN += (len(entities_pos) - current_TP)
        TP += current_TP

    print('TP：', TP)
    print('FP：', FP)
    print('FN：', FN)
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    try:
        F = 2 * P * R / (P + R)
    except:
        F = 0
    return P, R, F


def set_test(model, test_iter, session):

    if not test_iter.is_test:
        test_iter.is_test = True

    y_pred_list = []
    y_true_list = []
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length in tqdm.tqdm(
            test_iter):

        feed_dict = {
            model.input_x_word: input_ids_list,
            model.input_x_len: seq_length,
            model.input_relation: label_ids_list,
            model.input_mask: input_mask_list,

            model.keep_prob: 1,
            model.is_training: False,
        }

        lengths, logits, trans = session.run(
            fetches=[model.lengths, model.logits, model.trans],
            feed_dict=feed_dict
        )

        predict = decode(logits, lengths, trans)
        y_pred_list.append(predict)
        y_true_list.append(label_ids_list)

    y_pred_list = np.concatenate(y_pred_list)
    y_true_list = np.concatenate(y_true_list)

    precision, recall, f1 = get_P_R_F(y_pred_list, y_true_list)

    print('precision: {}, recall {}, f1 {}'.format(precision, recall, f1))

    return precision, recall


if __name__ == '__main__':
    config = Config()
    vocab_file= config.vocab_file  # 通用词典
    do_lower_case = False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    train_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'train.txt', use_bert=config.use_bert,
                              tokenizer=tokenizer, seq_length=config.sequence_length)

    dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'dev.txt', use_bert=config.use_bert, tokenizer=tokenizer,
                            seq_length=config.sequence_length, is_test=True)

    train(train_iter, dev_iter, config)
