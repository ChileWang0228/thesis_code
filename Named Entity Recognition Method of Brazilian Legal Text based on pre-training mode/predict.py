from train_fine_tune import decode
from config import Config
import tensorflow as tf
import os
import json
import numpy as np
from bert import tokenization
import tqdm
from utils import DataIterator
from train_fine_tune import get_P_R_F

result_data_dir = Config().new_data_process_quarter_final
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('GPU ID: ', str(gpu_id))
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Model: ', Config().checkpoint_path)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_session(checkpoint_path):
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            # Load the saved meta graph and restore variables
            try:
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_path))
            except OSError:
                saver = tf.train.import_meta_graph("{}.ckpt.meta".format(checkpoint_path))
            saver.restore(session, checkpoint_path)

            _input_x = graph.get_operation_by_name("input_x_word").outputs[0]
            _input_x_len = graph.get_operation_by_name("input_x_len").outputs[0]
            _input_mask = graph.get_operation_by_name("input_mask").outputs[0]
            _input_relation = graph.get_operation_by_name("input_relation").outputs[0]
            _keep_ratio = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            _is_training = graph.get_operation_by_name('is_training').outputs[0]


            used = tf.sign(tf.abs(_input_x))
            length = tf.reduce_sum(used, reduction_indices=1)
            lengths = tf.cast(length, tf.int32)
            logits = graph.get_operation_by_name('project/pred_logits').outputs[0]

            trans = graph.get_operation_by_name('transitions').outputs[0]

            def run_predict(feed_dict):
                return session.run([logits, lengths, trans], feed_dict)

    print('recover from: {}'.format(checkpoint_path))
    return run_predict, (_input_x, _input_x_len, _input_mask, _input_relation, _keep_ratio, _is_training)

def set_test(test_iter, model_file):
    if not test_iter.is_test:
        test_iter.is_test = True

    y_pred_list = []
    y_true_list = []
    predict_fun, feed_keys = get_session(model_file)
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, seq_length in tqdm.tqdm(test_iter):

        logits, lengths, trans = predict_fun(
            dict(
                zip(feed_keys, (input_ids_list, seq_length, input_mask_list, label_ids_list, 1, False))
                 )
        )

        pred = decode(logits, lengths, trans)
        y_pred_list.append(pred)
        y_true_list.append(label_ids_list)


    """
    融合所需参数保存
    """
    if 'test' in dev_iter.data_file:
        result_detail_f = 'test_result_detail_{}.txt'.format(config.checkpoint_path.split('/')[-1])
    else:
        result_detail_f = 'dev_result_detail_{}.txt'.format(config.checkpoint_path.split('/')[-1])

    with open(config.ensemble_source_file + result_detail_f, 'w', encoding='utf-8') as detail:
        for idx in range(len(y_pred_list)):
            item = {}
            item['pred'] = y_pred_list[idx]
            item['true'] = y_true_list[idx]
            detail.write(json.dumps(item, ensure_ascii=False, cls=NpEncoder) + '\n')

    y_pred_list = np.concatenate(y_pred_list)
    y_true_list = np.concatenate(y_true_list)

    precision, recall, f1 = get_P_R_F(y_pred_list, y_true_list)

    print('precision: {}, recall {}, f1 {}'.format(precision, recall, f1))


if __name__ == '__main__':
    config = Config()
    vocab_file= config.vocab_file
    do_lower_case =False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting test.txt..........')
    dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'test.txt', use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    # print('Predicting dev.txt..........')
    # dev_iter = DataIterator(config.batch_size, data_file=result_data_dir + 'dev.txt', use_bert=config.use_bert,
    #                         seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)

    set_test(dev_iter, config.checkpoint_path)
