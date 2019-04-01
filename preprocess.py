# -*- coding:utf-8 -*-
import jieba
import re
import os
import pickle
import logging  # 引入logging模块
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
# 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上

class Preprocess():
    __PAD__ = 0
    __GO__ = 1
    __EOS__ = 2
    __UNK__ = 3
    vocab = {'__PAD__':0, '__GO__':1, '__EOS__':2, '__UNK__':3}
    def __init__(self):
        self.save_dir = "data"
        self.dialog_dir = "dialog"
        self.Q_vocab = self.vocab.copy()
        self.A_vocab = self.vocab.copy()
        self.Q_vec = []
        self.A_vec = []
        self.START_ID = 4
        self.min_freq = 0
        self.use_char = True
        self.data_map = {}
 
    def main(self):
       
        with open(os.path.join(self.dialog_dir, "Q")) as Q_file:
            Qs = [i.strip() for i in Q_file.readlines()]
            self.get_vocab("Q", Qs)
            self.to_vec(self.Q_vocab,self.Q_vec,Qs)

        with open(os.path.join(self.dialog_dir, "A")) as A_file:
            As = [i.strip() for i in A_file.readlines()]
            self.get_vocab("A", As)
            self.to_vec(self.A_vocab,self.A_vec, As)

        # save 
        self.data_map = {
            "Q_vocab":self.Q_vocab,
            "Q_vec":self.Q_vec,
            "Q_vocab_size":max(self.Q_vocab.values()),
            "A_vocab":self.A_vocab,
            "A_vec":self.A_vec,
            "A_vocab_size":max(self.A_vocab.values()),
        }
        
        with open(os.path.join(self.save_dir, "map.pkl"),"wb") as f:
            pickle.dump(self.data_map, f)
        logging.info( "Q_vocab_size {}".format(self.data_map['Q_vocab_size']))
        logging.info( "A_vocab_size {}".format(self.data_map['A_vocab_size']))

    def get_vocab(self,dtype, sentences):
        words_count = {}
        if dtype == "Q":
            vocab = self.Q_vocab
        else:
            vocab = self.A_vocab
  
        for sent in sentences:
            if self.use_char:
                segments = list(sent)
            else:
                segments = jieba.lcut(sent)

            for seg in segments:
                if seg not in words_count:
                    words_count[seg] =  1
                else:
                    words_count[seg] += 1

        sorted_list = [[v[1], v[0]] for v in words_count.items()]
        sorted_list.sort(reverse=True)
        for index, item in enumerate(sorted_list):
            word = item[1]
            if item[0] < self.min_freq:
                break
            vocab[word] = self.START_ID + index
            #id2word_dict[self.START_ID + index] = word

        # save vocab 
        with open(os.path.join(self.save_dir, dtype+"_vocab"), "w") as f:
            for k,v in vocab.items():
                f.write("{},{}\n".format(k.encode("utf-8"),v))
        logging.info('_vocab save done!')

    def to_vec(self, vocab, vec,sentences):

        for sent in sentences:
            if self.use_char:
                segments = list(sent)
            else:
                segments = jieba.lcut(sent)
            temp_vec = []
            for seg in segments:
                if seg not in vocab:
                    temp_vec.append(vocab['__UNK__'])
                else:
                    temp_vec.append(vocab[seg])
            if vocab == self.A_vocab:
                temp_vec.append(vocab['__EOS__'])

            vec.append(temp_vec)
        logging.info('to_vec  done!')

def main():

    p = Preprocess()
    p.main()
if __name__ == '__main__':
    main()