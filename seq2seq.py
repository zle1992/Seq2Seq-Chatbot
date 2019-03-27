# -*- coding:utf-8 -*-
import pickle
import os
import re
import sys
import time
import jieba
import numpy as np
import tensorflow as tf
from dynamic_seq2seq_model import DynamicSeq2seq
from utils import BatchManager

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)
np.random.seed(1)
tf.set_random_seed(1)

import logging  # 引入logging模块
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
# 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上

class Seq2seq():
    '''
    tensorflow-1.0.0

        args:
        encoder_vec_file    encoder向量文件  
        decoder_vec_file    decoder向量文件
        encoder_vocabulary  encoder词典
        decoder_vocabulary  decoder词典
        model_path          模型目录
        batch_size          批处理数
        sample_num          总样本数
        max_batches         最大迭代次数
        show_epoch          保存模型步长

    '''
    def __init__(self):
        print("tensorflow version: ", tf.__version__)
        

        self.data_map = "data/map.pkl"
        self.batch_size = 256
        self.max_epoch = 1000
        self.lr = 0.001
        self.lstm_dims = 64
        self.beam_width = 3
        self.show_batch = 10
        self.model_path = 'model/'
        self.user_char = True
        # 获取输入输出
        try:
            with open(self.data_map, "rb") as f: 
                data_map = pickle.load(f)
        except IOError:
            logging.error('data_map not found!')

        self.encoder_vocab = data_map["Q_vocab"]
        self.encoder_vec = data_map["Q_vec"] #  id of word 
        self.encoder_vocab_size = data_map["Q_vocab_size"]
        self.char_to_vec = self.encoder_vocab
        
        self.decoder_vocab = data_map["A_vocab"]
        self.decoder_vec = data_map["A_vec"] #  id of word 
        self.decoder_vocab_size = data_map['A_vocab_size']

        self.vec_to_char = {v:k for k,v in self.decoder_vocab.items()}

        logging.info( "encoder_vocab_size {}".format(self.encoder_vocab_size))
        logging.info("decoder_vocab_size {}".format(self.decoder_vocab_size))

        self.model = DynamicSeq2seq(
            encoder_vocab_size=self.encoder_vocab_size+1,
            decoder_vocab_size=self.decoder_vocab_size+1,
            lr = self.lr,
            lstm_dims = self.lstm_dims,
            beam_width = self.beam_width,
        )
        self.sess = tf.Session()
        self.restore_model()
        
    def restore_model(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt:
            logging.info(ckpt.model_checkpoint_path)
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())
            logging.info("没找到模型")

    def get_fd(self, batch, model):
        '''获取batch

            为向量填充PAD    
            最大长度为每个batch中句子的最大长度  
            并将数据作转换:  
            [batch_size, time_steps] -> [time_steps, batch_size]
        '''
        encoder_inputs = batch[0]
        decoder_targets = batch[1]
        feed_dict = {
            model.encoder_inputs:encoder_inputs,
            model.decoder_targets:decoder_targets
        }
        return feed_dict

    def train(self):
        batch_manager = BatchManager(self.encoder_vec, self.decoder_vec, self.batch_size)

        loss_track = []
        total_time = 0
        nums_batch = len(batch_manager.batch_data)
        for epoch in range(self.max_epoch):
            print ("[->] epoch {}".format(epoch))   
            batch_index = 0
            for batch in batch_manager.batch():
                batch_index += 1
                # 获取fd [time_steps, batch_size]
                fd = self.get_fd(batch, self.model)
                _, loss, logits, labels,lr = self.sess.run([self.model.train_op, 
                                    self.model.loss,
                                    self.model.logits,
                                    self.model.decoder_labels,
                                    self.model.learning_rate
                                    ], fd)
                loss_track.append(loss)
                if batch_index % self.show_batch == 0:
                    print("\tstep: {}/{}".format(batch_index, nums_batch)+'\tloss: {}'.format(loss)+'\t lr:{}'.format(lr))
                    print("\t"+"-"*50)
                
                if len(loss_track) > 5 and loss > max(loss_track[-5:]):
                    sess.run(self.model.learning_rate_decay_op)
                

                checkpoint_path = self.model_path+"chatbot_seq2seq.ckpt"
                # 保存模型
                self.model.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step)
        
    def make_inference_fd(self, vec):
        tensor = np.array([vec])
        feed_dict = {
            self.model.encoder_inputs:tensor
        }
        return feed_dict

    def predict(self, input_str):
        if self.user_char:
            segments = list(input_str)
        else:
            segments = jieba.lcut(input_str)

        vec = [self.char_to_vec.get(seg, 3) for seg in segments]
        feed = self.make_inference_fd(vec)
        logits = self.sess.run([self.model.translations], feed_dict=feed)
        #print(logits)
        for i in range(self.beam_width):
            output = logits[0][0][:,i]

            output_str = "".join([self.vec_to_char.get(i, "_UN_") for i in output])

            print ("AI> "+output_str)


    def format_output(self,output_str, input_str):
        '''
        后处理
        '''
        return output_str

if __name__ == '__main__':
    if sys.argv[1]:
        seq = Seq2seq()
        if sys.argv[1] == 'train':
            seq.train()
        elif sys.argv[1] == 'infer':
            while True:
                
                sys.stdout.write("me> ")
                sys.stdout.flush()
                input_seq = sys.stdin.readline()
                seq.predict(input_seq) 
