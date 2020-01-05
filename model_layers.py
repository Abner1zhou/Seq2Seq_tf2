# -*- coding: UTF-8 –*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com 
@desc: Encoder, Decoder, Attention
"""
import tensorflow as tf

from utils.wv_loader import load_embedding_matrix, load_vocab


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_sz, emb_dim, emb_matrix, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.units = enc_units
        self.emb = tf.keras.layers.Embedding(vocab_sz, emb_dim,
                                             weights=[emb_matrix],
                                             trainable=False)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.emb(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def init_hidden(self):
        return tf.zeros((self.batch_sz, self.units))


class BahAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahAttention, self).__init__()
        # 全连接层
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, ht, hs):
        """
        :param ht: 上一个时间布decoder 输出的state   shape:(batch_size, decoder_units)
        :param hs: encoder的output   shape:(batch_size, sentence_max_length, encoder_units)
        :return: context vector， attention weight
        """
        # expend_ht shape:(batch_size, 1, decoder_units)
        expand_ht = tf.expand_dims(ht, 1)
        score = self.V(tf.nn.tanh(self.W1(hs), self.W2(expand_ht)))
        # attention weight  shape:(batch_size, sentence_max_length, 1)
        aw = tf.nn.softmax(score, axis=1)
        context_vector = aw * hs
        # context_vector shape:(batch_size, units)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, aw


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_sz, emb_dim, emb_matrix, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.units = dec_units
        self.emb = tf.keras.layers.Embedding(vocab_sz, emb_dim,
                                             weights=[emb_matrix],
                                             trainable=False)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, context_vector):
        # x shape:(batch_size, 1, embedding_dim)
        x = self.emb(x)
        # 把attention计算得到的上下文权重和上一个时间布Decoder的预测结果加在一起，作为本次GRU的输入
        # x shape:(batch_size, 1, embedding_dim + hidden_size)
        # axis=-1  等同 axis=2
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # output shape:(batch_size, 1, units)
        output, state = self.gru(x)
        # output去掉 axis=1,  reshape:(batch_size, units)
        output = tf.reshape(output, (-1, output.shape[2]))

        # prediction shape == (batch_size, vocab)
        prediction = self.fc(output)

        return prediction, state


if __name__ == '__main__':
    embedding_matrix = load_embedding_matrix()
    vocab, reverse_vocab = load_vocab()
    vocab_size = len(vocab)

    input_sequence_len = 250
    BATCH_SIZE = 64
    embedding_dim = 200
    units = 512
    # 测试Encoder, encoder直接输入一整句话，自动循环得到输出
    encoder = Encoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)
    example_input_batch = tf.ones(shape=(BATCH_SIZE, input_sequence_len), dtype=tf.int32)
    sample_hidden = encoder.init_hidden()

    sample_output, sample_state = encoder(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
    # 打印结果
    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)
    # 单步decoder， decoder每次预测一个词
    sample_decoder_output, _, = decoder(tf.random.uniform((64, 1)),
                                        attention_result)
    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
