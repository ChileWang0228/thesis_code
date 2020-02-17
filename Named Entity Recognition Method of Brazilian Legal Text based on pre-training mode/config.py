class Config:
    
    def __init__(self):
        
        self.embed_dense = True
        self.embed_dense_dim = 512
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.9
        self.relation_num = 7 + 1


        self.decay_rate = 0.5
        self.decay_step = 5000
        self.num_checkpoints = 20 * 3

        self.train_epoch = 20
        self.sequence_length = 128

        self.learning_rate = 1e-4
        self.embed_learning_rate = 5e-5
        self.batch_size = 20



        # 多语种
        self.bert_file = '/data/wangzhili/pretrained_model/multi_cased_L-12_H-768_A-12/bert_model.ckpt'
        self.bert_config_file = '/data/wangzhili/pretrained_model/multi_cased_L-12_H-768_A-12/bert_config.json'
        self.vocab_file = '/data/wangzhili/pretrained_model/multi_cased_L-12_H-768_A-12/vocab.txt'

        # 模型存放路径
        self.model_dir = '/data/wangzhili/BR_NER/model'
        # predict.py ensemble.py get_ensemble_final_result.py post_ensemble_final_result.py的结果路径
        self.continue_training = False
        self.ensemble_source_file  = '/data/wangzhili/BR_NER/ensemble/source_file/'
        self.ensemble_result_file = '/data/wangzhili/BR_NER/ensemble/result_file/'
        # 数据预处理的结果路径
        self.new_data_process_quarter_final = '/data/wangzhili/BR_NER/clean_csv_data/'

        # 模型
        # BILSTM
        # self.checkpoint_path = "/data/wangzhili/BR_NER/model/runs_3/1576478146/model_0.9259_0.9479-1818"
        # test: precision: 0.9195893926432849, recall 0.9442248572683355, f1 0.9317443120260022

        self.checkpoint_path = "/data/wangzhili/BR_NER/model/runs_4/1576478236/model_0.9069_0.9361-2424"
        # test: precision: 0.9075738125802311, recall 0.9302631578947368, f1 0.9187784275503573

        # IDCNN
        # self.checkpoint_path = "/data/wangzhili/BR_NER/model/runs_5/1576487630/model_0.8826_0.9191-606"
        # test: precision: 0.8678844519966016, recall 0.9350114416475973, f1 0.9001982815598149

        self.checkpoint_path = "/data/wangzhili/BR_NER/model/runs_6/1576487731/model_0.9192_0.9434-3838"
        # test: precision: 0.912617220801364, recall 0.9444199382443759, f1 0.9282462605679601



        self.model_type = 'idcnn'
        # self.model_type = 'bilstm'
        self.lstm_dim = 128
        self.dropout = 0.5
        self.use_origin_bert = True # True:使用原生bert, False:使用动态融合bert

