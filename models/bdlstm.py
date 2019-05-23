import tensorflow as tf
import sys
sys.path.append('..')
from flags import FLAGS


class RecurrentModel:
    def __init__(self):
        self.lstm_layers = FLAGS.rnn_num_layers
        self.num_hidden = FLAGS.rnn_cell_size

    def construct_model(self, x):
        y = self.build_lstm(x)
        return y, self.compute_loss(y)

    def build_lstm(self, x):
        lstm = tf.nn.rnn_cell.MultiRNNCell([self.get_lstm() for _ in range(self.lstm_layers)])
        output, state = tf.nn.static_rnn(cell=lstm, inputs=x, dtype=tf.float32)
        output = output[-1]
        return output

    def get_lstm(self):
        return tf.nn.rnn_cell.LSTMCell(self.num_hidden)

    def compute_loss(self, y):
        print('foo')
