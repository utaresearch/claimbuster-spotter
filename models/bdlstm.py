import tensorflow as tf
import sys
sys.path.append('..')
from flags import FLAGS


class RecurrentModel:
    def __init__(self):
        pass

    def construct_model(self, x, x_len, y, embed):
        yhat = self.build_lstm(x, x_len, embed)
        return yhat, self.compute_loss(y, yhat)

    def build_lstm(self, x, x_len, embed):
        x = tf.unstack(x, axis=1)
        for i in range(len(x)):
            x[i] = tf.nn.embedding_lookup(embed, x[i])
        x = tf.stack(x, axis=1)

        lstm = tf.nn.rnn_cell.MultiRNNCell([self.get_lstm() for _ in range(FLAGS.rnn_num_layers)])
        output, state = tf.nn.dynamic_rnn(cell=lstm, inputs=x, sequence_length=x_len, dtype=tf.float32)

        print(output)
        exit()

        add_weight = tf.get_variable('post_lstm_weight', shape=(FLAGS.rnn_cell_size, FLAGS.num_classes),
                                     initializer=tf.contrib.layers.xavier_initializer())
        add_bias = tf.get_variable('post_lstm_bias', shape=FLAGS.num_classes,
                                   initializer=tf.contrib.layers.xavier_initializer())

        print(output[-1])
        return tf.matmul(output[-1], add_weight) + add_bias

    @staticmethod
    def get_lstm():
        return tf.nn.rnn_cell.LSTMCell(FLAGS.rnn_cell_size)

    @staticmethod
    def compute_loss(y, yhat):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=yhat)
