# -*- coding:utf-8 -*-
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers import safe_embedding_lookup_sparse as embedding_lookup_unique
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleLSTMCell
from tensorflow.contrib.framework import nest
import logging  # 引入logging模块
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')  # logging.basicConfig函数对日志的输出格式及方式做相关配置
# 由于日志基本配置中级别设置为DEBUG，所以一下打印信息将会全部显示在控制台上

class DynamicSeq2seq():
    '''
    Dynamic_Rnn_Seq2seq with Tensorflow-1.9.0  

        args:
        encoder_cell            encoder结构
        decoder_cell            decoder结构
        encoder_vocab_size      encoder词典大小
        decoder_vocab_size      decoder词典大小
        embedding_size          embedd成的维度
        bidirectional           encoder的结构
                                True:  encoder为双向LSTM 
                                False: encoder为一般LSTM
        attention               decoder的结构
                                True:  使用attention模型
                                False: 一般seq2seq模型
                      控制输入数据格式
                                True:  [time_steps, batch_size]
                                False: [batch_size, time_steps]

        
    '''
    PAD = 0
    EOS = 2
    UNK = 3
    def __init__(self, 
                encoder_vocab_size=10,
                decoder_vocab_size=5, 
                lr = 1,
                embedding_size=128,
                lstm_dims = 56,
                beam_width = 0,
                attention=False,
                debug=False,
                time_major=False):
        
        self.debug = debug
        self.attention = attention
        self.lstm_dims = lstm_dims
        self.init_lr = lr
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.beam_width = beam_width
        self.embedding_size = embedding_size

        
        
        
        self.global_step = tf.Variable(-1, trainable=False)
        self.max_gradient_norm = 5

        #创建模型
        self._make_graph()

    def _make_graph(self):
        # 创建占位符
        self._init_placeholders()

        # embedding层
        self._init_embeddings()

        # 判断是否为双向LSTM并创建encoder
        self._init_bidirectional_encoder()

        # 创建decoder，会判断是否使用attention模型
        self._init_decoder()

        # 计算loss及优化
        self._init_optimizer()

    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.batch_size = tf.shape(self.encoder_inputs)[0]
        self.decoder_inputs = tf.concat([tf.ones(shape=[self.batch_size, 1], dtype=tf.int32), self.decoder_targets], 1)
        self.decoder_labels = tf.concat([self.decoder_targets, tf.zeros(shape=[self.batch_size, 1], dtype=tf.int32)], 1)

        used = tf.sign(tf.abs(self.encoder_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.encoder_inputs_length = tf.cast(length, tf.int32)

        used = tf.sign(tf.abs(self.decoder_labels))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.decoder_targets_length = tf.cast(length, tf.int32) 


        self.learning_rate = tf.Variable(float(self.init_lr), trainable=False, dtype=tf.float32)

        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.9)



    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:

            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            # encoder Embedding
            embedding_encoder = tf.get_variable(
                    "embedding_encoder", 
                    shape=[self.encoder_vocab_size, self.embedding_size],
                    initializer=initializer,
                    dtype=tf.float32
                )
            self.encoder_emb_inp = tf.nn.embedding_lookup(
                    embedding_encoder, self.encoder_inputs
                )
            #  decoder Embedding
            embedding_decoder = tf.get_variable(
                    "embedding_decoder", 
                    shape=[self.decoder_vocab_size, self.embedding_size],
                    initializer=initializer,
                    dtype=tf.float32
                )
            self.embedding_decoder = embedding_decoder
            self.decoder_emb_inp = tf.nn.embedding_lookup(
                    embedding_decoder, self.decoder_inputs
                )
            
    def _init_bidirectional_encoder(self):
        '''
        双向LSTM encoder
        '''
        # Build RNN cell
        #encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_dims)

        encoder_cell = CudnnCompatibleLSTMCell(self.lstm_dims)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, self.encoder_emb_inp,
            sequence_length=self.encoder_inputs_length, time_major=False,
            dtype=tf.float32
        )

        ''''

        todo  :s双向rnn
        '''
        # Construct forward and backward cells
        # forward_cell = CudnnCompatibleLSTMCell(self.lstm_dims)
        # backward_cell =CudnnCompatibleLSTMCell(self.lstm_dims)

        # bi_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
        #     forward_cell, backward_cell, self.encoder_emb_inp,
        #     sequence_length=self.encoder_inputs_length, time_major=False,dtype=tf.float32)
        # encoder_outputs = tf.concat(bi_outputs, -1)



        self.encoder_output = encoder_outputs
        self.encoder_state = encoder_state

    def _init_decoder(self):


        self.decoder_cell=CudnnCompatibleLSTMCell(self.lstm_dims)


        attention_states = self.encoder_output

        with tf.variable_scope('shared_attention_mechanism'):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.lstm_dims, 
                memory=attention_states,
            )
            

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            self.decoder_cell, attention_mechanism,
            attention_layer_size=self.lstm_dims
        )

        # Helper    
        helper = tf.contrib.seq2seq.TrainingHelper(
            self.decoder_emb_inp, 
            self.decoder_targets_length+1, 
            time_major=False
        )

        projection_layer = tf.layers.Dense(self.decoder_vocab_size, use_bias=False)
        init_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_state)
              
        # Decoder

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cell,
            helper=helper,
            initial_state=init_state,
            output_layer=projection_layer
        )


        maximum_iterations = tf.round(tf.reduce_max(self.encoder_inputs_length) * 20)
        # Dynamic decoding


        with tf.variable_scope('decode_with_shared_attention'):

            self.outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder, 
                maximum_iterations=maximum_iterations,
            )
        self.logits = self.outputs[0][0]




        # ------------Infer-----------------




        if self.beam_width>1:

            tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                self.encoder_output, multiplier=self.beam_width)

            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                self.encoder_state, multiplier=self.beam_width)

            with tf.variable_scope('shared_attention_mechanism', reuse=True):
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                        num_units=self.lstm_dims, 
                        memory=tiled_encoder_outputs,
                    )
            decoder_cell  = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell, attention_mechanism,
                attention_layer_size=self.lstm_dims,
            )
            decoder_initial_state = decoder_cell.zero_state(
                dtype=tf.float32, batch_size=self.batch_size * self.beam_width)
            decoder_initial_state = decoder_initial_state.clone(
                cell_state=tiled_encoder_final_state)
              

             # Define a beam-search decoder
            inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=self.embedding_decoder,
                    start_tokens=tf.fill([self.batch_size], 1),
                    end_token=2,
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_width,
                    output_layer=projection_layer,
                    )



        else:
            # decoder阶段根据是否使用beam_search决定不同的组合，
            # 如果使用则直接调用BeamSearchDecoder（里面已经实现了helper类）
            # 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码

            # Helper
            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.embedding_decoder,
                tf.fill([self.batch_size], 1), 2)
        
            # Decoder
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=infer_helper,
                initial_state=init_state,
                output_layer=projection_layer
                )


         # Dynamic decoding
        with tf.variable_scope('decode_with_shared_attention', reuse=True):
            infer_outputs= tf.contrib.seq2seq.dynamic_decode(
                inference_decoder, maximum_iterations=maximum_iterations,)




        # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
        # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
        # rnn_output: [batch_size, decoder_targets_length, vocab_size]
        # sample_id: [batch_size, decoder_targets_length], tf.int32

        # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
        # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
        # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
        # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果



        if self.beam_width>1:
            self.translations = infer_outputs[0].predicted_ids
            
        else:
            self.translations = tf.expand_dims(infer_outputs[0].sample_id,-1)
            

    def _init_optimizer(self):
        # 整理输出并计算loss
        mask = tf.sequence_mask(
            tf.to_float(self.decoder_targets_length),
            tf.to_float(tf.shape(self.decoder_labels)[1])    
        )

        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.logits,
            self.decoder_labels,
            tf.to_float(mask)
        )

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
                        gradients, self.max_gradient_norm)

        # Optimization
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        self.train_op = update_step
        self.saver = tf.train.Saver(tf.global_variables())

    def run(self):
        feed = {
            self.encoder_inputs:[[2,1],[1,2],[2,3],[3,4],[4,5]],
            self.decoder_targets:[[1,1],[1,1],[4,1],[3,1],[2,0]],
        }
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                logits,_,loss = sess.run([self.outputs, self.train_op, self.loss], feed_dict=feed)