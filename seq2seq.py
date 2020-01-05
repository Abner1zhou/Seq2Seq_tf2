# -*- coding: UTF-8 –*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com 
@desc: 把各层穿起来
"""
import tensorflow as tf

from model_layers import Encoder, BahAttention, Decoder
from utils.params_utils import  VOCAB_SIZE, EMBEDDING_DIM, UNITS, BATCH_SIZE
from utils.wv_loader import load_vocab, load_embedding_matrix


class Seq2Seq(tf.keras.Model):
    def __init__(self, vocab_sz, emb_dim,  enc_units, dec_units, batch_sz):
        super(Seq2Seq, self).__init__()
        self.embedding_matrix = load_embedding_matrix()

        self.encoder = Encoder(vocab_sz, emb_dim,
                               self.embedding_matrix, enc_units,
                               batch_sz)

        self.attention = BahAttention(enc_units)

        self.decoder = Decoder(vocab_sz, emb_dim,
                               self.embedding_matrix, dec_units,
                               batch_sz)

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.init_hidden()
        enc_output, enc_state = self.encoder(enc_inp, enc_hidden)
        return enc_output, enc_state

    def call_decoder_one_step(self, dec_input, dec_state, enc_output):
        context_vector, attention_weights = self.attention(dec_state, enc_output)
        prediction, dec_state = self.decoder(dec_input, context_vector)
        return prediction, dec_state, context_vector, attention_weights

    def call(self, dec_input, dec_hidden, enc_output, dec_target):
        predictions = []
        # attentions = []
        context_vector, _ = self.attention(dec_hidden, enc_output)

        for t in range(1, dec_target.shape[1]):
            pred, dec_hidden = self.decoder(dec_input,
                                            context_vector)
            context_vector, attn = self.attention(dec_hidden, enc_output)
            # Teacher Forcing
            dec_input = tf.expand_dims(dec_target[:, t], 1)

            predictions.append(pred)
            # attentions.append(attn)

        return tf.stack(predictions, 1), dec_hidden


def main():
    vocab, re_vocab = load_vocab()
    vocab_size = len(vocab)
    batch_size = 128
    input_sequence_len = 200

    model = Seq2Seq(vocab_size, EMBEDDING_DIM, UNITS, UNITS, batch_size)

    # example_input
    example_input_batch = tf.ones(shape=(batch_size, input_sequence_len), dtype=tf.int32)

    # sample input
    sample_hidden = model.encoder.init_hidden()

    sample_output, sample_hidden = model.encoder(example_input_batch, sample_hidden)

    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahAttention(10)
    context_vector, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    sample_decoder_output, _, = model.decoder(tf.random.uniform((batch_size, 1)),
                                              context_vector)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))


if __name__ == '__main__':
    main()
