import tensorflow as tf
import os

x = tf.placeholder(tf.int32, (None, None), name='x')
y = tf.placeholder(tf.int32, (None,), name='y')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'



if __name__ == '__main__':
    main()