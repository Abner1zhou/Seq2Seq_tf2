# -*- coding: UTF-8 –*-
"""
@author: 周世聪
@contact: abner1zhou@gmail.com 
@desc: 训练
"""
import tensorflow as tf

from seq2seq import Seq2Seq
from train_helper import train_model
from utils.wv_loader import load_vocab
from utils.params_utils import TRAIN_EPOCH, BATCH_SIZE, EMBEDDING_DIM, UNITS
from utils.config import checkpoint_dir
from utils.gpu_config import config_gpu


def train():
    # 设置GPU
    config_gpu()

    # 读取vocab训练
    vocab, reverse_vocab = load_vocab()
    vocab_size = len(vocab)

    # 构建模型
    print("Building the model ...")
    model = Seq2Seq(vocab_size, EMBEDDING_DIM, UNITS, UNITS, BATCH_SIZE)

    # 获取保存管理者
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    # 训练模型
    train_model(model, vocab, TRAIN_EPOCH, BATCH_SIZE, checkpoint_manager)


if __name__ == '__main__':
    train()
