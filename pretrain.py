import tensorflow as tf
import os
from data_utils.data_loader import DataLoader

x = tf.placeholder(tf.int32, (None, None), name='x')
y = tf.placeholder(tf.int32, (None,), name='y')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_load = DataLoader()
    train_data = data_load.load_training_data()
    validation_data = data_load.load_validation_data()


if __name__ == '__main__':
    main()