# -*- coding: UTF-8 –*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com 
@desc: 制作数据读入生成器
"""

from utils.data_processing import load_train_data, load_test_data
from utils.config import *

import tensorflow as tf


def train_batch_generator(batch_size, sample_sum=None):
    """
    加载训练集
    :param batch_size:
    :param sample_sum: 校验集大小
    :return: 训练集和步长
    """
    train_x, train_y = load_train_data()
    if sample_sum:
        train_x = train_x[:sample_sum]
        train_y = train_y[:sample_sum]
    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(len(train_x))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    steps_per_epoch = len(train_x) // batch_size
    return dataset, steps_per_epoch


def bream_test_batch_generator(beam_size, max_enc_len=200):
    # 加载数据集
    test_x = load_test_data(max_enc_len)
    for row in test_x:
        beam_search_data = tf.convert_to_tensor([row for i in range(beam_size)])
        yield beam_search_data
