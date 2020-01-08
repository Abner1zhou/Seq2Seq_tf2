# -*- coding: UTF-8 –*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com 
@desc: 构造训练魔性
"""
import tensorflow as tf
import time

from data_loader import train_batch_generator


def train_model(model, vocab, epochs, batch_sz, checkpoint_manager):
    vocab_size = len(vocab)
    pad_index = vocab['<PAD>']
    start_index = vocab['<START>']
    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=0.01)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # 定义损失函数
    def loss_function(real, pred):
        # 计算train label真实长度，把pad置0，其他置1  例如：（1,1,1,1,0,0,0,0)
        mask = tf.math.logical_not(tf.math.equal(real, pad_index))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def train_step(enc_inp, dec_target):
        with tf.GradientTape() as tape:
            enc_out, enc_state = model.call_encoder(enc_inp)
            # 第一个decoder输入
            dec_inp = tf.expand_dims([start_index] * batch_sz, 1)
            # 第一个decoder隐藏层输入
            dec_hidden = enc_state
            # 预测输出
            predictions, _ = model(dec_inp, dec_hidden, enc_out, dec_target)

            batch_loss = loss_function(dec_target[:, 1:], predictions)
            variables = model.encoder.trainable_variables + model.decoder.trainable_variables + model.attention.trainable_variables
            gradients = tape.gradient(batch_loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss

    dataset, steps_per_epoch = train_batch_generator(batch_sz)

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0

        for (batch, (inputs, target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inputs, target)
            total_loss += batch_loss

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))




